#####################
Scaling your training
#####################

.. warning::
    To scale your training or **run more than 1 simulation** model at a time (in parallel) you will need an **exported version** of your AnyLogic model. In order for you to *export the model*, you **MUST** have a **valid AnyLogic Professional license**.

If you are using the ``rllib`` package, you can define how many simulation instances you want to execute in parallel by tunning the **rollouts** parameters:

.. code-block:: python

    from alpyperl.examples.cartpole_v0.cartpole_env import CartPoleEnv
    from ray.rllib.algorithms.ppo import PPOConfig

    # Initialise policy configuration (e.g. PPOConfig), rollouts and environment
    policy = (
        PPOConfig()
        .rollouts(
            num_rollout_workers=2,
            num_envs_per_worker=5,
        )
        .environment(
            CartPoleEnv, 
            env_config={
                'run_exported_model': False,
                'exported_model_loc': './resources/exported_models/cartpole_v0'
                'show_terminals': False
            }
        )
        .build()
    )

    # Create training loop
    for _ in range(10):
        result = policy.train()

    # Save policy at known location
    checkpoint_dir = policy.save("./resources/trained_policies/cartpole_v0")
    print(f"Checkpoint saved in directory '{checkpoint_dir}'")

    # Close all enviornments (otherwise AnyLogic model will be hanging)
    policy.stop()

The total number of simulation instances will be defined by ``num_rollout_workers x num_envs_per_worker``.

In addition to that, you must enable ``'run_exported_model': True`` in the **environment** and point to the exported model folder ``'exported_model_loc': './resources/exported_models/cartpole_v0'`` in order for the parallel execution to take place.
