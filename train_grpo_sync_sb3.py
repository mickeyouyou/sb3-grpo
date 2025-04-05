import gymnasium as gym
import highway_env
import numpy as np

from envs import configure_env, configure_eva_env, record_videos,CustomSyncVectorEnv
from grpo_sb3 import GRPO

from stable_baselines3.common.callbacks import ProgressBarCallback

class MyProgressBarCallback(ProgressBarCallback):
    def __init__(self):
        super(MyProgressBarCallback, self).__init__()
    def _on_step(self) -> bool:
        # You can add custom code here to perform actions at each step
        # For example, you can log additional information or modify the training process
        # self.pbar.update(self.)
        return True


def train():
    group_size=10
    env = CustomSyncVectorEnv([configure_env for _ in range(group_size)])
    model = GRPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="highway_grpo_sb3/"
    )
    # model = GRPO.load("models/grpo_sb3_highway", env, device='cuda')

    model.learn(total_timesteps=500_000, progress_bar=True)
    model.save("models/grpo_sb3_highway")

def test():
    env = configure_eva_env(render_mode='rgb_array')
    env = record_videos(env, "videos_grpo_0328_sb3")
    model = GRPO.load("models/grpo_sb3_highway", env, device='cpu')
    obs, _ = env.reset()
    while True:
        action, states_ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"action {action} speed {info['speed']:.2f}")
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    train()
    # test()