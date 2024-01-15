from alpyperl.serve.rllib import launch_policy_server
from alpyperl import AnyLogicEnv
from ray.rllib.algorithms.ppo import PPOConfig


# Launch server
launch_policy_server(
    policy_config=PPOConfig(),
    env=AnyLogicEnv,
    trained_policy_loc='./resources/trained_policies/cartpole_v2',
    port=3002
)