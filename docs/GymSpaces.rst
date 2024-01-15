######################
How to define a space
######################

*ALPypeRL* has been developed around the ``Gym.Env`` framework and is compatible with most of the ``gym.spaces``. The ``gym.spaces`` is a collection of tools to define the state and action spaces of a simulation model.

There are two ways to define **Action and Observation spaces** in *ALPypeRL*:

* Directly from the ``ALPypeRLConnector`` in *AnyLogic*.
* Using the ``BaseAnyLogicEnv`` class inheritance in *python*.

************************************************
1. Using the ``ALPypeRLConnector`` in *AnyLogic*
************************************************

If you choose to define the spaces directly from the ``ALPypeRLConnector`` in *AnyLogic*, you will need to define the them in the object properties. Beforehand, you must check **Create here (else define in python)**. Two section will then appear: ``Observation Space`` and ``Action Space``.

Both spaces use the same structure under the ``Builder`` concept. The steps to follow are:

* Initialize your *space builder* by either calling ``ActionSpace.init()`` or ``ObservationSpace.init()``.
* Add the desired spaces by calling ``.add(GymSpace space)`` concurrently.
* Build the space by calling ``.build()`` at the end.

.. note::
    A ``GymSpace`` is normally created by using the *static* constructors located at ``GymSpaces``. For example, to create a ``Discrete`` space you can use ``GymSpaces.discrete(int n)``. This will create a ``Discrete`` space with ``n`` values.

The ``gym.spaces`` library provides a wide range of spaces. You can find the full list of spaces in the `gym.spaces <https://gymnasium.farama.org/api/spaces/fundamental/>`_ documentation. The supported spaces, if you define them from *AnyLogic*, are:

---------
Discrete
---------

Supports a single discrete number of values with and optional start for the values:

.. code-block:: java

    // A single discrete number starting with 0
    GymSpaces.discrete(int n)

    // A single discrete number starting with custom number
    GymSpaces.discrete(int n, int startValue)


------------------
Box (Continuous)
------------------

Supports continuous vectors or matrices, used for vector observations, images, etc:

.. code-block:: java

    // A single continuous value
    GymSpaces.box(Double lowBound, Double upperBound)

    // A continuous vector
    GymSpaces.box(Double[] lowBoundArr, Double[] UpperBoundArr)

    // A continuous matrix
    GymSpaces.box(Double[][] lowBoundArr, Double[][] UpperBoundArr)

    // A continuous symetric vector
    GymSpaces.box(Double lowBound, Double upperBound, int nrows)

    // A continuous symetric matrix
    GymSpaces.box(Double lowBound, Double upperBound, int nrows, int ncols)

.. note::
    The box space uses ``Double`` instead of ``double`` because it also supports unbounded spaces. If you want to unbound a space, you can use ``GymSpaces.Box.unbounded()`` (e.g., ``GymSpaces.box(GymSpace.Box.unbounded(), 10.0)``).

------------
MultiBinary
------------

Suppors a vector of binary values, used for holding down a button or if an agent has an object:

.. code-block:: java

    // A vector of binary values
    GymSpaces.multiBinary(int n)

--------------
MultiDiscrete
--------------

Supports multiple discrete values with multiple axes, used for controller actions:

.. code-block:: java

    // A vector of discrete values
    GymSpaces.multiDiscrete(int[] nvec)

    // A vector of discrete values with custom start
    GymSpaces.multiDiscrete(int[] nvec, int[] startValues)

------
Tuple
------

Supports a tuple of subspaces, used for a fixed number of ordered spaces:

.. code-block:: java

    // A tuple of subspaces
    GymSpaces.tuple(GymSpace[] spaces)

    // A tuple of subspaces with custom names
    GymSpaces.tuple(List<GymSpace> spaces)

.. note::
    This space is specially useful if you require to combine **continuous** and **discrete** spaces. For example, if you want to combine a ``Discrete`` and a ``Box`` space, you can do it as follows:

    .. code-block:: java

        // A tuple of subspaces
        GymSpaces.tuple(List.of(GymSpaces.discrete(10), GymSpaces.box(0.0, 1.0)))

------
Dict 
------

Supports a dictionary of keys and subspaces, used for a fixed number of ordered spaces:

.. code-block:: java

    // A dictionary of keys and subspaces
    GymSpaces.dict(String name, GymSpace space)

    // A dictionary of keys and subspaces with custom names
    GymSpaces.dict(Map<String, GymSpace> spaces)

********************************************
2. Using the ``BaseAnyLogicEnv`` in *python*
********************************************

You can define spaces directly by using the ``gym.spaces``. Please refer to their documentation for further details. Remember that when you define the spaces from *python* you must inherit the ``BaseAnyLogicEnv`` class.

The steps are:

* Create your own custom class and inherit the ``BaseAnyLogicEnv`` class.
* Define the ``action_space`` and ``observation_space`` variables. Remember to set them as local by using the ``self.`` prefix (e.g. ``self.action_space``).
* Call the ``super(<YourCustomEnv>).__init__(env_config)`` method at the end of the constructor (e.g. ``super(CartPoleEnv).__init__(env_config)``).

.. code-block:: python
    :emphasize-lines: 7, 25, 24, 29

    import math
    from gymnasium import spaces
    import numpy as np
    from alpyperl import BaseAnyLogicEnv


    class CartPoleEnv(BaseAnyLogicEnv):

        def __init__(self, env_config=None):

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
            self.action_space = spaces.Discrete(2)
            self.observation_space = spaces.Box(-high, high, dtype=np.float32)
            
            # IMPORTANT: Initialise AnyLogic environment experiment after
            # environment creation
            super(CartPoleEnv, self).__init__(env_config)