from alpyperl.examples.cartpole_v3.cartpole_env import CartPoleEnv
from ray.rllib.algorithms.ppo import PPOConfig

# Set checkpoint directory.
checkpoint_dir = "./resources/trained_policies/cartpole_v3"

# Initialize policy.
policy = (
    PPOConfig()
    .env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=2
    )
    .fault_tolerance(
        recreate_failed_env_runners=True,
        num_consecutive_env_runner_failures_tolerance=3
    )
    .environment(
        CartPoleEnv, 
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v3_custom_env',
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
for _ in range(100):
    result = policy.train()

# Save policy checkpoint.
policy.save(checkpoint_dir)
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments.
policy.stop()