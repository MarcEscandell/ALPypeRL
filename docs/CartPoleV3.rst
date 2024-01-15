##############################################################
How to set an array of mixed actions. The CarPole-v3 example.
##############################################################

.. note::
  You may find the source code of the *CartPole-v3* `here <https://github.com/MarcEscandell/ALPypeRL/tree/main/alpyperl/examples/cartpole_v3/CartPole_v3>`_.

*CartPole-v3* simply demonstrates how to set an array of **mixed actions**. In this case, the action space is composed of two variables, one of them being a continuous variable and the other discrete. The first variable is a continuous force applied to the cart on the left, and the second one is a discrete force applied to the cart on the right.

You will have to modify exactly the same pieces of the model as you did for *v1*, resulting in:

* A new **action space** as shown in the code below:

  * In java/AnyLogic:

  .. code-block:: java

      ActionSpace.init()
        .add(GymSpaces.box(-1.0, 0.0))	        // Left force (continuous)
        .add(GymSpaces.discrete(2))		// Right force (discrete)
        .build()

  * In python/ALPypeRL:

  .. code-block:: python

      self.action_space = spaces.Tuple([
          spaces.Box(np.array([0, -1]), np.array([1, 0]), dtype=np.float32),
          spaces.Discrete(2)
      ])

* And a new first part of the ``takeAction(RLAction action)`` function body:

.. code-block:: java

    // Perform action
    cartPole.applyForce(action.getInt(1) + action.getDouble(0));

    // [...]