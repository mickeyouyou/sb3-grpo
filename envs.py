import numpy as np
import gymnasium as gym
import highway_env
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.vec_env import VecEnv

def record_videos(env, video_folder="dqn_videos"):
    wrapped = RecordVideo(
        env, video_folder=video_folder, episode_trigger=lambda e: True
    )

    # Capture intermediate frames
    env.unwrapped.set_record_video_wrapper(wrapped)

    return wrapped

def configure_env(render_mode=None):
    env = gym.make('highway-v0', render_mode=render_mode)
    # env = gym.make('custom-highway-v0', render_mode=render_mode)
    
    # 配置环境参数
    env.unwrapped.configure({
        'observation': {
            'type': 'Kinematics',
            'vehicles_count': 5,
            'features': ['presence', 'x', 'y', 'vx', 'vy', 'heading'],
        },
        'action': {
            'type': 'DiscreteMetaAction',
        },
        'lanes_count': 4,
        'vehicles_count': 15,
        'controlled_vehicles': 1,
        'duration': 40,  # 每个episode的持续时间（秒）
        'initial_spacing': 2,
        'simulation_frequency': 15,  # 模拟频率（Hz）
        'policy_frequency': 5,  # 决策频率（Hz）
        'screen_width': 600,  # 渲染宽度（像素）
        'screen_height': 150,  # 渲染高度（像素）
        'centering_position': [0.3, 0.5],
        'scaling': 5.5,
        'show_trajectories': True,
        'render_agent': True,
        'offscreen_rendering': False,
        # 'normalize_reward': False,
        "nearby_vehicle_reward": 0,
        "collision_reward": -1,  # The reward received when colliding with a vehicle.
        "right_lane_reward": 0,  # The reward received when driving on the right-most lanes, linearly mapped to
        # zero for other lanes.
        "high_speed_reward": 1,  # The reward received when driving at full speed, linearly mapped to zero for
        # lower speeds according to config["reward_speed_range"].
        "lane_change_reward": 0,  # The reward received at each lane change action.
    })
    
    # 重置环境
    obs, info = env.reset(seed=99)
    return env

# 创建并配置highway环境
def configure_eva_env(render_mode=None):
    env = configure_env(render_mode=render_mode)
    # 重置环境
    obs, info = env.reset(seed=9999)
    return env

# 自定义容器环境类
class CustomSyncVectorEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]  # 创建多个环境实例
        self.num_envs = len(self.envs)
        # 从第一个环境获取观测空间和动作空间
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        # 初始化 done 状态跟踪
        self.dones = np.zeros(self.num_envs, dtype=bool)
        obs_tuple = [env.reset() for env in self.envs]  # 初始观测
        self.obs = [ob[0] for ob in obs_tuple]  # 只保存观测值

    def step(self, actions):
        obs = []
        rewards = []
        dones = []
        truncateds = []
        infos = []

        # 对每个环境执行一步
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:  # 如果已经停止，则返回维度不变但值全为0的观测和零奖励
                # obs.append(self.obs[i])
                obs.append(np.zeros_like(self.obs[i]))
                rewards.append(0.0)
                dones.append(True)
                truncateds.append(False)
                infos.append({})
            else:
                ob, reward, terminated, truncated, info = env.step(action)
                obs.append(ob)
                rewards.append(reward)
                dones.append(terminated)
                truncateds.append(truncated)
                infos.append(info)
                self.obs[i] = ob  # 更新观测
                if terminated or truncated:
                    self.dones[i] = True  # 标记为已停止

        # 返回向量化结果
        return (np.array(obs), np.array(rewards), np.array(dones), np.array(truncateds), infos)

    def reset(self):
        # 重置所有环境，并清空 done 状态
        seed = np.random.randint(0, 1000)
        reset_results = [env.reset(seed=seed) for env in self.envs]
        self.obs = [result[0] for result in reset_results]  # 只保存观测值
        self.dones = np.zeros(self.num_envs, dtype=bool)
        return np.array(self.obs)

    def reset_if_all_done(self):
        # 检查是否所有环境都已停止，若是则重置
        if np.all(self.dones):
            return self.reset()
        return np.array(self.obs)

    def close(self):
        for env in self.envs:
            env.close()

    def env_is_wrapped(self, wrapper_class, indices=None):
        # 检查环境是否被指定的wrapper包装
        if indices is None:
            indices = range(self.num_envs)
        return [isinstance(self.envs[i], wrapper_class) for i in indices]

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        # 在指定环境上调用方法
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], method_name)(*method_args, **method_kwargs) for i in indices]

    def get_attr(self, attr_name, indices=None):
        # 获取指定环境的属性值
        if indices is None:
            indices = range(self.num_envs)
        return [getattr(self.envs[i], attr_name) for i in indices]

    def set_attr(self, attr_name, values, indices=None):
        # 设置指定环境的属性值
        if indices is None:
            indices = range(self.num_envs)
        for i, value in zip(indices, values):
            setattr(self.envs[i], attr_name, value)

    def step_async(self, actions):
        # 异步执行步骤（在同步实现中，只存储动作）
        self._actions = actions

    def step_wait(self):
        # 等待异步步骤完成并返回结果（在同步实现中，直接执行步骤）
        return self.step(self._actions)
    def __len__(self):
        return len(self.envs)
