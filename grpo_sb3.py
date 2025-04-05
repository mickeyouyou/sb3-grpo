from typing import (Any, Callable, ClassVar, Dict, Optional, Type, TypeVar,
                    Union)
import numpy as np
import torch
import copy
import sys
import time
from gymnasium import spaces
from highway_env import utils
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.policies import (ActorCriticCnnPolicy,
                                               ActorCriticPolicy, BasePolicy,
                                               MultiInputActorCriticPolicy)
from stable_baselines3.common.type_aliases import (GymEnv, MaybeCallback,
                                                   RolloutReturn, Schedule)
from stable_baselines3.common.utils import get_schedule_fn
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

SelfGRPO = TypeVar("SelfGRPO", bound="GRPO")

class GRPO(OnPolicyAlgorithm):
    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 200,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[Type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[Dict[str, Any]] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        reward_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ):
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {}
        if rollout_buffer_class is None:
            rollout_buffer_class = RolloutBuffer

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )
        
        self.group_size = 10
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.normalize_advantage = normalize_advantage
        self.clip_range = clip_range

        self.epsilon_low = 0.2
        self.epsilon_high = 0.8
        self.beta = 0.1

        self.samples_per_time_step = 5
        self.middle_steps = 5
        self.reward_function = reward_function 
        self.my_rollout_buffer_kwargs = rollout_buffer_kwargs

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        self._setup_lr_schedule()
        self.set_random_seed(self.seed)
        self.policy = self.policy_class(
            self.observation_space, self.action_space, self.lr_schedule, use_sde=self.use_sde, **self.policy_kwargs
        ).to(self.device)

        self.ref_policy = copy.deepcopy(self.policy)

        # Initialize schedules for clipping ranges
        self.clip_range = get_schedule_fn(self.clip_range)

        # Create a rollout buffer with a size accounting for samples per step
        buffer_size = self.n_steps * self.samples_per_time_step * self.n_envs
        self.rollout_buffer = self.rollout_buffer_class(
            buffer_size,
            self.observation_space,
            self.action_space,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            n_envs=self.n_envs,
            **self.my_rollout_buffer_kwargs,
        )

    def collect_rollouts(
        self,
        env,
        callback: MaybeCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> RolloutReturn:
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        rollout_buffer.reset()
         # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        episode_obs = [[] for _ in range(self.group_size)]
        episode_rewards = [[] for _ in range(self.group_size)]
        episode_infos = [[] for _ in range(self.group_size)]
        episode_actions = [[] for _ in range(self.group_size)]
        episode_log_probs = [[] for _ in range(self.group_size)]

        obs = env.reset()
        for step in range(n_rollout_steps):# max sample steps
            with torch.no_grad():
                obs_tensor = torch.from_numpy(obs).to(self.device)
                actions, values, log_probs = self.policy(obs_tensor)
            actions = actions.cpu().numpy()

            obs, rewards, dones, truncated, infos = env.step(actions)

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            for group in range(self.group_size):
                if infos[group]:
                    episode_obs[group].append(obs[group])
                    episode_rewards[group].append(rewards[group])
                    episode_infos[group].append(infos[group])
                    episode_actions[group].append(actions[group])
                    episode_log_probs[group].append(log_probs[group])

            if np.all(dones):
                # print(f"Finished Sample seqs from vector environments max step {step}")
                break

            # 当所有环境都停止时，手动重置
            obs = env.reset_if_all_done()
            self._last_episode_starts = dones
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                break
            self._update_info_buffer(infos, dones)

        for group in range(self.group_size):
            self.num_timesteps += len(episode_actions[group])

        self.episode_data =  {
            'states': episode_obs,
            'actions':episode_actions,
            'rewards': episode_rewards,
            'infos': episode_infos,
            'log_probs': log_probs,
        }
        self._last_obs = obs

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def compute_seq_rewards(self, rollout):
        rewards_array = np.array([0.0 for i in range(self.group_size)], dtype=np.float32)

        # compute reward for each group
        for g in range(self.group_size):
            g_info = rollout['infos'][g]
            crashed_cnt = np.count_nonzero([info['crashed'] for info in g_info])
            lane_chage_action_cnt = sum([action in (0, 2) for action in self.episode_data['actions'][g]])
            ad_chage_action_cnt = sum([action in (3, 4) for action in self.episode_data['actions'][g]])
            mean_speed = np.mean([info['speed'] for info in g_info])
            # speed_reward = utils.lmap(
            #     mean_speed,
            #     [25, 30],
            #     [0, 1],
            # )
            speed_reward = 1 if 20 <= mean_speed and mean_speed <= 30 else 0
            comfort_reward = -lane_chage_action_cnt / len(g_info) - ad_chage_action_cnt / len(g_info)

            # rewards_array[g] = speed_reward - crashed_cnt
            rewards_array[g] = 1 - crashed_cnt

        rewards = torch.from_numpy(rewards_array).to(self.device)
        return rewards
    
    def calculate_relative_advantages(self, rewards):
        mean_rewards = rewards.mean()
        # if mean_rewards == 1:
        #     return torch.ones_like(rewards)
        # elif mean_rewards == 0:
        #     return torch.ones_like(rewards) * -1
        # std_rewards = rewards.std() + 1e-8
        
        # relative_advantages = (rewards - mean_rewards) / std_rewards
        relative_advantages = rewards - mean_rewards
        return relative_advantages
    
    def train(self):
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        rewards = self.compute_seq_rewards(self.episode_data)
        relative_advantages = self.calculate_relative_advantages(rewards)
        print("rewards", " ".join(f"{r:.2f}" for r in rewards.tolist()))
        print("advantage", " ".join(f"{a:.2f}" for a in relative_advantages.tolist()))

        # 执行多次梯度更新，每次都重新计算损失
        for i in range(self.n_epochs):
            loss_list = []
            g_kl_loss = []
            clip_fractions = []
            # 因为每组序列长度不同，分开过策略网络
            for g in range(self.group_size):
                states_array = np.array(self.episode_data['states'][g], dtype=np.float32)
                states_g = torch.from_numpy(states_array).to(self.device).reshape(-1, 30)
                actions_array = np.array(self.episode_data['actions'][g], dtype=np.long)
                actions_g = torch.from_numpy(actions_array).to(self.device)

                reward_array = np.array(self.episode_data['rewards'][g], dtype=np.float32)
                reward_g = torch.from_numpy(reward_array).to(self.device)

                # 旧策略的动作概率
                old_log_probs = self.episode_data['log_probs'][g]

                with torch.no_grad():
                    ref_actions, ref_values, ref_log_probs = self.ref_policy(states_g)

                # 计算新的动作概率
                values_, new_log_probs, entrpy_ = self.policy.evaluate_actions(states_g, actions_g)

                # 计算策略比率和裁剪后的目标函数
                ratio = torch.exp(new_log_probs - old_log_probs)
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                surr1 = ratio * reward_g.T* relative_advantages[g]
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_low, 1.0 + self.epsilon_high) * reward_g.T * relative_advantages[g]
                g_loss = -torch.min(surr1, surr2)

                # Calculate KL divergence as per formula (2)
                # D_KL(π_θ||π_ref) = π_ref(o_i|q)/π_θ(o_i|q) - log(π_ref(o_i|q)/π_θ(o_i|q)) - 1
                ratio_kl = torch.exp(new_log_probs - ref_log_probs)
                kl_loss = (ratio_kl - torch.log(ratio_kl) - 1)

                loss_list.append((g_loss + self.beta* kl_loss).sum())

            loss = torch.stack(loss_list).mean()
            # 更新网络
            self.policy.optimizer.zero_grad()
            loss.backward()
            # 添加梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=self.max_grad_norm)
            # 执行优化步骤
            self.policy.optimizer.step()
            self._n_updates += 1

        # only for compute rollouts staticstics
        min_g = float('inf')
        max_g = 0
        sum_episode = 0
        for group in range(self.group_size):
            sum_episode += len(self.episode_data['actions'][group])
            min_g = min(len(self.episode_data['actions'][group]), min_g)
            max_g = max(len(self.episode_data['actions'][group]), max_g)
        # Logs
        # self.logger.record("train/policy_gradient_loss", loss.item())
        # self.logger.record("train/kl_loss", kl_loss.item())
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        self.logger.record("train/group_timestep_min", min_g)
        self.logger.record("train/group_timestep_max", max_g)
        self.logger.record("train/group_timestep_mean", sum_episode/self.group_size)
        # self.logger.record("train/rewards", rewards.tolist())
        # self.logger.record("train/relative_advantage", relative_advantages.tolist())

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))
        self.logger.dump(step=self.num_timesteps)


    def learn(
        self: SelfGRPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "GRPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfGRPO:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            self.ref_policy.load_state_dict(self.policy.state_dict())
            for middle_step in range(self.middle_steps):
                self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

                iteration += 1
                self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

                # Display training infos
                if log_interval is not None and iteration % log_interval == 0:
                    assert self.ep_info_buffer is not None
                    self.dump_logs(iteration)

                self.train()

        callback.on_training_end()

        return self