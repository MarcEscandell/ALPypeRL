#################################################################
Running your simulation model as an AnyLogic ``CustomExperiment``
#################################################################

The ``CustomExperiment`` is a specific AnyLogic experiment type, enabling developers to generate and control experiments **solely through code**, without utilizing a graphical interface or predefined settings. For instance, in a *Monte-Carlo experiment*, users can define custom expressions for the simulation model parameters for each iteration and the output value to display in the results histogram.

This experiment type is commonly employed by developers to run the model without any visual elements (no simulation window will be launched), enabling interaction with the model exclusively through the *Console*. This approach is particularly advantageous when the primary focus is on **performance** or on running as many iterations as possible in a minimal time frame. This is facilitated by the absence of visual element rendering.

.. warning::
    However, **ALPypeRL** requires you to notify the ``ALPypeRLConnector`` that it is running under a ``CustomExperiment`` and pass the required information (which include the *experiment command line arguments*).

    **At the moment, ALPypeRL is unable to collect this information by itself.**

Here's a guide on how to proceed in case of the :ref:`CartPole-v0 example<How to train your first policy. The CartPole-v0 example.>`:

.. code-block:: java
    :emphasize-lines: 17

    // Create Engine, initialize random number generator:
    Engine engine = createEngine();
    engine.setTimeUnit( SECOND );
    // Fixed seed (reproducible simulation runs)
    engine.getDefaultRandomGenerator().setSeed( System.currentTimeMillis() );
    engine.setStartTime( 0.0 );
    engine.setStartDate( toDate( 2023, FEBRUARY, 28, 16, 0, 0 ) );
    // Set stop time:
    engine.setStopTime( 2000.0 );
    // Create new root object:
    Main root = new Main( engine, null, null );
    // Setup parameters of root object here
    root.setParametersToDefaultValues();
    // Set RL experiment mode to training
    root.rlMode = RLMode.TRAIN;
    // Notify ALPypeRLConnector is running under a custom experiment
    root.alPypeRLConnector.isCustomExperiment(getCommandLineArguments());
    // Prepare Engine for simulation:
    engine.start( root );

.. note::
    If you need to, do remember to set the *rlMode* as ``RLMode.TRAIN`` as shown in the example.

    This is required in the package **examples** as the mode is set via an additional parameter. If you define this parameter directly at ``ALPypeRLConnector`` then you do not need to specify its value in the code.

If you want to test a ``CustomExperiment`` or use it as a base model for your project, you can find it at:

* The experiment has been created and stored inside the ``.alp`` file for the **CartPole-v0** example at ``./alpyperl/examples/cartpole_v0/CartPole_v0``.
* There is also an exported version located at ``./resources/exported_models/cartpole_v0_custom_experiment``. Remember to point to this folder in your ``train.py`` script ``env_config`` and ``exported_model_loc`` when defining your environment.



