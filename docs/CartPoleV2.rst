##################################################################
How to set an array of continuous actions. The CarPole-v2 example.
##################################################################

.. note::
  You may find the source code of the *CartPole-v2* `here <https://github.com/MarcEscandell/ALPypeRL/tree/main/alpyperl/examples/cartpole_v2/CartPole_v2>`_.

*CartPole-v2* example is a continuation of *v1*. In this case, your action space will be defined as an array of size 2 (which you can grow to any `n` size to suit your specific problem). The value of the indices refer to whether the force is applied from *left* (index ``<0``) or *right* (index ``>0``) and the intensity of each.

You will have to modify exactly the same pieces of the model as you did for *v1*, resulting in:

* A new **action space** as shown in the code below:

  * In java/AnyLogic:

  .. code-block:: java

      ActionSpace.init()
        .add(GymSpaces.box(-1.0, 0.0))
        .add(GymSpaces.box(0.0, 1.0))
        .build()

  * In python:

  .. code-block:: python

      self.action_space = spaces.Box(np.array([0, -1]), np.array([1, 0]), dtype=np.float32)

* And a new first part of the ``takeAction(RLAction action)`` function body:

.. code-block:: java

    // Perform action
    cartPole.applyForce(action.getDouble(1) + action.getDouble(0));

    // [...]

.. warning::
  The *CartPole-v2* example described here is only used for the purpose of showing **how an array of actions** can be set. Processing 2 actions in the way it is done in this example does not make much sense, as a single continuous variable would do the job.