# from utils import record_videos, show_videos
import matplotlib.cm as cm
import matplotlib as mpl
import importlib
import gymnasium as gym
import highway_env
import numpy as np
import sys
from tqdm.notebook import trange
import os
import pygame

from envs import configure_env, configure_eva_env, record_videos, CustomSyncVectorEnv
from rl_agents.agents.common.factory import agent_factory
# from rl_agents.agents.dynamic_programming.graphics import ValueIterationGraphics

sys.path.insert(0, './highway-env/scripts/')


class MyValueIterationGraphics(object):
    """
        Graphical visualization of the Value Iteration value function.
    """
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    highway_module = None

    @classmethod
    def display(cls, agent, surface):
        """
            Display the computed value function of an agent.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        # """
        # if not cls.highway_module:
        #     try:
        #         cls.highway_module = importlib.import_module("highway_env")
        #     except ModuleNotFoundError:
        #         pass
        # if cls.highway_module and isinstance(agent.env, cls.highway_module.envs.common.abstract.AbstractEnv):
        cls.display_highway(agent, surface)

    @classmethod
    def display_highway(cls, agent, surface):
        """
            Particular visualization of the state space that is used for highway_env environments only.

        :param agent: the agent to be displayed
        :param surface: the surface on which the agent is displayed.
        """
        import pygame
        norm = mpl.colors.Normalize(vmin=-2, vmax=2)
        cmap = cm.jet_r
        try:
            grid_shape = agent.mdp.original_shape
        except AttributeError:
            grid_shape = cls.highway_module.finite_mdp.compute_ttc_grid(
                agent.env, time_quantization=1., horizon=10.).shape
        cell_size = (surface.get_width(
        ) // grid_shape[2], surface.get_height() // (grid_shape[0] * grid_shape[1]))
        speed_size = surface.get_height() // grid_shape[0]
        value = agent.get_state_value().reshape(grid_shape)
        for h in range(grid_shape[0]):
            for i in range(grid_shape[1]):
                for j in range(grid_shape[2]):
                    color = cmap(norm(value[h, i, j]), bytes=True)
                    pygame.draw.rect(surface, color, (
                        j * cell_size[0], i * cell_size[1] + h * speed_size, cell_size[0], cell_size[1]), 0)
            pygame.draw.line(surface, cls.BLACK,
                             (0, h * speed_size), (grid_shape[2] * cell_size[0], h * speed_size), 1)
        states, actions = agent.plan_trajectory(agent.mdp.state)
        for state in states:
            (h, i, j) = np.unravel_index(state, grid_shape)
            pygame.draw.rect(surface, cls.RED,
                             (j * cell_size[0], i * cell_size[1] + h * speed_size, cell_size[0], cell_size[1]), 1)



env = gym.make("highway-v0", render_mode="rgb_array")
# 配置环境参数
env.unwrapped.configure({
    'observation': {
        'type': 'TimeToCollision',
        'horizon': 10,
    },
    # 'duration': 60,  # 每个episode的持续时间（秒）‘
    # 'vehicles_count': 70,
#     # 'initial_spacing': 2,
#     # 'simulation_frequency': 15,  # 模拟频率（Hz）
#     # 'policy_frequency': 5,  # 决策频率（Hz）
#     # 'screen_width': 600,  # 渲染宽度（像素）
#     # 'screen_height': 150,  # 渲染高度（像素）
#     'centering_position': [0.3, 0.5],
#     'scaling': 5.5,
#     'show_trajectories': True,
#     'offscreen_rendering': False,
})
env = record_videos(env)
(obs, info), done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "< class 'rl_agents.agents.dynamic_programming.value_iteration.ValueIterationAgent'>",
    "iterations": 1000,
    "gamma": 0.9,
}
agent = agent_factory(env, agent_config)

pygame.init()
surface = pygame.display.set_mode((600, 500))
# 分出上下区域
env_surface = surface.subsurface(pygame.Rect(0, 0, 600, 150))
value_surface = surface.subsurface(pygame.Rect(0, 150, 600, 350))
# Run episode
for step in trange(env.unwrapped.config["duration"], desc="Running..."):
    action = agent.act(obs)
    obs, reward, done, truncated, info = env.step(action)

    MyValueIterationGraphics.display(agent, value_surface)
    pygame.display.flip()
    pygame.image.save(value_surface, f"steps/value_function_{step}.png")

env.close()
pygame.quit()
# show_videos()
