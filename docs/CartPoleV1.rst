#########################################################
How to set continuous actions. The CartPole-v1 example.
#########################################################

.. note::
  You may find the source code of the *CartPole-v1* `here <https://github.com/MarcEscandell/ALPypeRL/tree/main/alpyperl/examples/cartpole_v1/CartPole_v1>`_.

The *CartPole-v1* model is very similar to *v0*. However, this time, instead of only allowing two 2 actions (``0`` or ``1``, representing *right* or *left*) you are going to define a single range of values that goes from anywere between ``-1`` and ``1``. Such change will impact the orginal model in the following parts:

* **Observation space**. Now the ``spaces.Discrete(2)`` instance is no longer valid. To create a continuous range, you have to use ``spaces.Box``. So you will be replacing your ``self.action_space`` in your ``CartPoleEnv(BaseAnyLogicEnv)`` to:

.. code-block:: python
  
    self.action_space = spaces.Box(-1, 1, dtype=np.float32)

* ``takeAction(ActionSpace Action)`` function implementation in **AnyLogic**. You will now be changing the first part of the body to:

.. code-block:: java

    // Perform action
    cartPole.applyForce(action.getDoubleAction());

    // [...]

Everything else remains untouched.