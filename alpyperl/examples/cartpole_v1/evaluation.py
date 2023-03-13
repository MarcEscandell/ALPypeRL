from alpyperl.serve.rllib.binder import launch_policy_server
from alpyperl.examples.cartpole_v1.cartpole_env import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig


# Launch server
launch_policy_server(
    policy_config=PPOConfig(),
    env=CartPoleEnv,
    trained_policy_loc='./resources/trained_policies/cartpole_v1/checkpoint_000011',
    port=3000
)