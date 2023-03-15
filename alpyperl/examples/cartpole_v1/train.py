from alpyperl.examples.cartpole_v1 import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig


policy = (
    PPOConfig()
    .rollouts(
        num_rollout_workers=2,
        num_envs_per_worker=2,
        ignore_worker_failures=True,
        recreate_failed_workers=True,
        num_consecutive_worker_failures_tolerance=3
    )
    .environment(
        CartPoleEnv, 
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v1',
            'show_terminals': False
        }
    )
    .build()
)

for _ in range(11):
    result = policy.train()

checkpoint_dir = policy.save("./resources/trained_policies/cartpole_v1")
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments
policy.stop()