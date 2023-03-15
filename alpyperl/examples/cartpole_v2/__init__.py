from alpyperl.examples.cartpole_v2 import CartPoleEnv
import gymnasium as gym
from gymnasium import register


register(
    id='ALCartPole-v2',
    entry_point='alpyperl.examples.cartpole_v2.cartpole_env:CartPoleEnv',
    max_episode_steps=2000,
    disable_env_checker=True
)