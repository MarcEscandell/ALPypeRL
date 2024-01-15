import pytest
from alpyperl import AnyLogicEnv
from ray.rllib.algorithms.ppo import PPOConfig
import logging
logger = logging.getLogger(__name__)


@pytest.mark.order(1)
@pytest.mark.parametrize("example_index", [0, 1, 2, 3])
def test_run_example_train(example_index):
    logger.info("\nRunning example train for cartpole_v{}".format(example_index))
    # Set checkpoint directory
    checkpoint_dir = f"./tests/trained_policies/cartpole_v{example_index}"
    # Initialize policy
    policy = (
        PPOConfig()
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=2
        )
        .environment(
            AnyLogicEnv, 
            env_config={
                'run_exported_model': True,
                'exported_model_loc': f'./resources/exported_models/cartpole_v{example_index}',
                'show_terminals': False,
                'verbose': True,
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
    # Perform training
    for _ in range(20):
        result = policy.train()
    # Save policy checkpoint
    policy.save(checkpoint_dir)
    logger.info(f"Checkpoint saved in directory '{checkpoint_dir}'")
    # Close all enviornments
    policy.stop()