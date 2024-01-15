from alpyperl.examples.cartpole_v1.cartpole_env import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig

# Set checkpoint directory.
checkpoint_dir = "./resources/trained_policies/cartpole_v1"

# Initialize policy.
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
            'exported_model_loc': './resources/exported_models/cartpole_v1_custom_env',
            'show_terminals': False,
            'verbose': False,
            'checkpoint_dir': checkpoint_dir,
            'env_params': {
                'cartMass': 1.0,
                'poleMass': 0.1,
                'poleLength': 0.5,
            }
        }
    )
    .build()
)

# Perform training.
for _ in range(10):
    result = policy.train()

# Save policy checkpoint.
policy.save(checkpoint_dir)
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments.
policy.stop()