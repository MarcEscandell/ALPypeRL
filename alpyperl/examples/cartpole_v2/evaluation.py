from alpyperl.serve.rllib.binder import launch_policy_server
from alpyperl.examples.cartpole_v2.cartpole_env import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig


# Launch server
launch_policy_server(
    policy_config=PPOConfig(),
    env=CartPoleEnv,
    trained_policy_loc='./resources/trained_policies/cartpole_v2/checkpoint_000012',
    port=3000
)