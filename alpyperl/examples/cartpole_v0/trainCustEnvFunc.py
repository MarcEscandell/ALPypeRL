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
        create_custom_env(action_space, observation_space), 
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v0',
            'show_terminals': False,
            'verbose': False
        }
    )
    .build()
)

for _ in range(10):
    result = policy.train()

checkpoint_dir = policy.save("./resources/trained_policies/cartpole_v0")
print(f"Checkpoint saved in directory '{checkpoint_dir}'")

# Close all enviornments
policy.stop()