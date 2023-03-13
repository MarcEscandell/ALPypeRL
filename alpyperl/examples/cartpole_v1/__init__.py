import gymnasium as gym
from gymnasium import register


register(
    id='ALCartPole-v1',
    entry_point='examples.cartpole_v1.cartpole_env:CartPoleEnv',
    max_episode_steps=2000,
    disable_env_checker=True
)