from ray.rllib.algorithms.ppo import PPOConfig
import math
from gymnasium import spaces
import numpy as np
from alpyperl import create_custom_env


# The cartpole ACTION and OBSERVATION spaces
# ------------------------------------------

# Positional thresholds
theta_threshold_radians = 12 * 2 * math.pi / 360.0
x_threshold = 2.4
# Create observation space array thresholds
high = np.array(
    [
        x_threshold * 2,            # Horizontal position
        np.finfo(np.float32).max,   # Linear speed
        theta_threshold_radians * 2,# Pole angle
        np.finfo(np.float32).max    # Angular velocity
    ]
)
# Create Action and Observation spaces using `gymnasium.spaces`
action_space = spaces.Discrete(2)
observation_space = spaces.Box(-high, high, dtype=np.float32)


# The TRAIN script
# ------------------------------------------

# Set checkpoint directory.
checkpoint_dir = "./resources/trained_policies/cartpole_v0_custom_env"

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
        create_custom_env(action_space, observation_space), 
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v0_custom_env',
            'show_terminals': False,
            'verbose': False,
            
            'env_params': {
                'cartMass': 1.0,
                'poleMass': 0.1,
                'poleLength': 0.5,
            }
        }
    )
    .build()
)

for _ in range(100):
    result = policy.train()

checkpoint_dir = policy.save(checkpoint_dir)
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments
policy.stop()