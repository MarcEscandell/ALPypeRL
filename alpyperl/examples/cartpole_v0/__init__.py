from alpyperl.examples.cartpole_v0.cartpole_env import CartPoleEnv
from gymnasium import register


register(
    id='ALCartPole-v0',
    entry_point='alpyperl.examples.cartpole_v0.cartpole_env:CartPoleEnv',
    max_episode_steps=2000,
    disable_env_checker=True
)