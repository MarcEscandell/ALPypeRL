#################################################################
How to set simulation parameter values from your training script
#################################################################

You are allowed to set parameter values in your AnyLogic simulation from your python training script. Simply define them in the ``env_config`` and ``env_param``:

.. code-block:: python
    :emphasize-lines: 26, 27, 28, 29

    from alpyperl import AnyLogicEnv
    from ray.rllib.algorithms.ppo import PPOConfig

    # Set checkpoint directory.
    checkpoint_dir = "./resources/trained_policies/cartpole_v0"

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
            AnyLogicEnv, 
            env_config={
                'run_exported_model': True,
                'exported_model_loc': './resources/exported_models/cartpole_v0',
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

.. important::
    You **must comply** with the following conditions:

    * The parameter name must be the same as the parameter name in the AnyLogic model.
    * The parameter types suported are only primitive types. (e.g. int, float, bool, etc.)