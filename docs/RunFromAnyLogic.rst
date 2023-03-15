##########################################
Running your model directly from AnyLogic
##########################################

If you are **debugging** or **unable to export** your model (e.g. you don't have a valid license) you can run your simulation experiment directly from *AnyLogic* and connect to your *python script*.

To do so, you should take into account the following considerations:

* In your *python script*, regardless of the framework that you use, you should ensure that:

    * You add the ``run_exported_model`` into the ``env_config`` dictionary and set to ``False``.
    * Your framework **only launches 1 instance** of the environment. For example, in ``rllib`` that can be set by defining ``num_rollout_workers=1`` and ``num_envs_per_worker=1`` in **rollouts** section of ``AlgorithmConfig`` (e.g. ``PPOConfig``).

* You execute first your *python script* and then launch your *AnyLogic* model. When you run your python script, you will receive a message like this:

    .. code-block:: console

        (RolloutWorker pid=6909) 2023-03-17 06:37:36,639 [alpyperl.anylogic.model.connector][    INFO] You can now launch your AnyLogic model! 'ALPypeRLConnector' will handle the connection for you.

  After that, you are good to proceed to *AnyLogic*.

This is an example of how your python script could look like:

.. code-block:: python
    :emphasize-lines: 8, 9, 14

    from alpyperl.examples.cartpole_v0 import CartPoleEnv
    from ray.rllib.algorithms.ppo import PPOConfig


    policy = (
        PPOConfig()
        .rollouts(
            num_rollout_workers=1,
            num_envs_per_worker=1,
        )
        .environment(
            CartPoleEnv, 
            env_config={
                'run_exported_model': False,
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