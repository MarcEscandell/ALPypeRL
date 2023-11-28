from alpyperl.examples.cartpole_v2 import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig


policy = (
    PPOConfig()
    .rollouts(
        num_rollout_workers=2,
        num_envs_per_worker=2
    )
    .fault_tolerance(
        recreate_failed_workers=True,
        num_consecutive_worker_failures_tolerance=3
    )
    .environment(
        CartPoleEnv, 
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v2',
            'show_terminals': False,
            'verbose': False
        }
    )
    .build()
)

# Perform training
for _ in range(100):
    result = policy.train()

# Save checkpoint
checkpoint_dir = "./resources/trained_policies/cartpole_v2/checkpoint_000010"
policy.save(checkpoint_dir)
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments
policy.stop()