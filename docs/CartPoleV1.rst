#########################################################
How to set continuous actions. The CartPole-v1 example.
#########################################################

.. note::
  You may find the source code of the *CartPole-v1* `here <https://github.com/MarcEscandell/ALPypeRL/tree/main/alpyperl/examples/cartpole_v1/CartPole_v1>`_.

The *CartPole-v1* model is very similar to *v0*. However, this time, instead of only allowing two 2 actions (``0`` or ``1``, representing *right* or *left*) you are going to define a single range of values that goes from anywere between ``-1`` and ``1``. Such change will impact the orginal model in the following parts:

* **Observation space**. Now the ``ActionSpace.init().add(GymSpaces.discrete(2)).build()`` (in java/AnyLogic) or ``spaces.Discrete(2)`` (in python) instance is no longer valid. To create a continuous range, you have to use ``GymSpaces.Box`` (in java/AnyLogic) or ``spaces.Box`` (in python):


  * In java/AnyLogic:

  .. code-block:: java

      ActionSpace.init().add(GymSpaces.box(-1, 1)).build();

  * In python:

  .. code-block:: python
  
      self.action_space = spaces.Box(-1, 1, dtype=np.float32)

* ``takeAction(RLAction Action)`` function implementation in **AnyLogic**. You will now be changing the first part of the body to:

.. code-block:: java

    // Perform action
    cartPole.applyForce(action.getDoubleAction());

    // [...]

Everything else remains untouched.