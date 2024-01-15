from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
import numpy as np
from py4j.java_gateway import is_instance_of, get_java_class, java_import
from py4j.java_collections import JavaList
import pickle
import os
from filelock import FileLock


def get_anylogic_rl_action(anylogic_model, flattened_action, anylogic_action_space):
    """Initialize AnyLogic 'RLAction' from given flattened action array and AnyLogic 'ActionSpace'"""
    # Create consumable by AnyLogic side 'RLAction' given flattened action array.
    # AnyLogic action space is needed in order for AnyLogic to perform unflatten operation.
    return anylogic_model.jvm.com.alpype.RLAction(
        __get_java_array(anylogic_model, flattened_action),
        anylogic_action_space
    )

def parse_gym_to_anylogic_rl_space(anylogic_model, action_space=None, observation_space=None):
    """Parse gymnasium.spaces to AnyLogic 'ActionSpace' or 'ObservationSpace'"""
    # Parse gym.spaces to AnyLogic consumable 'ActionSpace' or 'ObservationSpace' by using
    # available ALPype Java API RLSpace builders.
    if action_space is not None:
        return anylogic_model.jvm.com.alpype.ActionSpace.init() \
                .add(__convert_gym_space_to_anylogic(anylogic_model, action_space)) \
                .build()
    elif observation_space is not None:
        return anylogic_model.jvm.com.alpype.ObservationSpace.init() \
                .add(__convert_gym_space_to_anylogic(anylogic_model, observation_space)) \
                .build()

def parse_anylogic_rl_space(anylogic_model, anylogic_rl_space):
    """Parse AnyLogic 'ActionSpace' or 'ObservationSpace' to gymnasium.spaces equivalent"""
    # Parse AnyLogic 'ActionSpace' or 'ObservationSpace' to gym.spaces
    # First, check size of space
    space_size = anylogic_rl_space.size()
    # If space size is 1, only parse single space
    if space_size == 1:
        return __convert_anylogic_space_to_gym(anylogic_model, anylogic_rl_space.get(0))
    # If space size is greater than 1, parse multiple spaces and include them in 
    # 'spaces.Tuple'
    else:
        return spaces.Tuple(
            [__convert_anylogic_space_to_gym(anylogic_model, anylogic_rl_space.get(i)) for i in range(space_size)]
        )

def __convert_gym_space_to_anylogic(anylogic_model, space):
    """[INTERNAL] Convert gym.spaces to AnyLogic Java 'GymSpace' equivalent"""
    if isinstance(space, spaces.Discrete):
        return anylogic_model.jvm.com.alpype.GymSpaces.discrete(int(space.n), int(space.start))
    elif isinstance(space, spaces.Box):
        if flatdim(space) == 1:
            return anylogic_model.jvm.com.alpype.GymSpaces.box(float(space.low[0]), float(space.high[0]))
        elif flatdim(space) > 1 and len(space.shape) == 1:
            return anylogic_model.jvm.com.alpype.GymSpaces.box(
                __get_java_array(anylogic_model, space.low.flatten(), 'Double'), 
                __get_java_array(anylogic_model, space.high.flatten(), 'Double')
            )
        # TODO: Find a way to handle 2D box spaces
        elif flatdim(space) > 1 and len(space.shape) == 2:
            return anylogic_model.jvm.com.alpype.GymSpaces.box(
                float(space.low[0, 0]), float(space.high[0, 0]),
                int(space.shape[0]), int(space.shape[1])
            )
    elif isinstance(space, spaces.MultiBinary):
        return anylogic_model.jvm.com.alpype.GymSpaces.multibinary(int(space.n))
    elif isinstance(space, spaces.MultiDiscrete):
        return anylogic_model.jvm.com.alpype.GymSpaces.multidiscrete(
            __get_java_array(anylogic_model, space.nvec, 'int'),
            __get_java_array(anylogic_model, space.start, 'int')
        )
    elif isinstance(space, spaces.Tuple):
        # Create java array using 'py4j' given data type and array length
        jarray_spaces = anylogic_model.new_array(anylogic_model.jvm.com.alpype.GymSpace, len(space.spaces))
        # Populate array with values from action and cast them accordingly
        for i, s in enumerate(space.spaces):
            jarray_spaces[i] = __convert_gym_space_to_anylogic(anylogic_model, s)
        # Construct AnyLogic 'GymSpaces.Tuple' from Java array
        return anylogic_model.jvm.com.alpype.GymSpaces.tuple(jarray_spaces)
    elif isinstance(space, spaces.Dict):
        # Create java map using 'py4j' given data type
        jmap_spaces = anylogic_model.jvm.java.util.LinkedHashMap()
        # Populate array with values from action and cast them accordingly
        for k, s in space.spaces.items():
            jmap_spaces.put(k, __convert_gym_space_to_anylogic(anylogic_model, s))
        # Construct AnyLogic 'GymSpaces.Tuple' from Java array
        return anylogic_model.jvm.com.alpype.GymSpaces.dict(jmap_spaces)

    raise Exception(f"Unsupported space type: {type(space)}")

def __convert_anylogic_space_to_gym(anylogic_model, anylogic_space):
    """[INTERNAL] Convert AnyLogic Java 'GymSpace' to gym.spaces equivalent"""
    if is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.Discrete):
        return spaces.Discrete(n=anylogic_space.sampleSize())
    elif is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.Box):
        # Construct gym.spaces.Box as a matrix
        if (
            anylogic_space.getNumRows() is not None and anylogic_space.getNumCols() is not None
            and anylogic_space.getNumRows() > 1 and anylogic_space.getNumCols() > 1
        ):
            return spaces.Box(
                low=np.array([
                    [
                        float(anylogic_space.lb(row, col)) if anylogic_space.lb(row, col) is not None else -np.inf
                        for col in range(anylogic_space.getNumCols())
                    ]
                    for row in range(anylogic_space.getNumRows())
                ]),
                high=np.array([
                    [
                        float(anylogic_space.ub(row, col)) if anylogic_space.ub(row, col) is not None else np.inf
                        for col in range(anylogic_space.getNumCols())
                    ] 
                    for row in range(anylogic_space.getNumRows())
                ])
            )
        # Construct gym.spaces.Box as a vector
        elif (
            anylogic_space.getNumRows() is not None and anylogic_space.getNumCols() is not None
            and anylogic_space.getNumRows() == 1 and anylogic_space.getNumCols() > 1
        ):
            return spaces.Box(
                low=np.array([
                    float(anylogic_space.lb(col)) if anylogic_space.lb(col) is not None else -np.inf
                    for col in range(anylogic_space.getNumCols())
                ]),
                high=np.array([
                    float(anylogic_space.ub(col)) if anylogic_space.ub(col) is not None else np.inf
                    for col in range(anylogic_space.getNumCols())
                ])
            )
        # By default, construct gym.spaces.Box as a single value
        return spaces.Box(
            low=anylogic_space.lb() if anylogic_space.lb() is not None else -np.inf,
            high=anylogic_space.ub() if anylogic_space.ub() is not None else np.inf
        )
    elif is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.MultiBinary):
        return spaces.MultiBinary(n=anylogic_space.spaceSize())
    elif is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.MultiDiscrete):
        return spaces.MultiDiscrete(nvec=list(anylogic_space.getVector()))
    elif is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.Tuple):
        return spaces.Tuple(
            spaces=[
                __convert_anylogic_space_to_gym(anylogic_model, anylogic_space.getSpace(i)) 
                for i in range(anylogic_space.spaceSize())
            ]
        )
    elif is_instance_of(anylogic_model, anylogic_space, anylogic_model.jvm.com.alpype.GymSpaces.Dict):
        return spaces.Dict(
            spaces={
                k: __convert_anylogic_space_to_gym(anylogic_model, anylogic_space.getSpace(k)) 
                for k in list(anylogic_space.getSpaceNames())
            }
        )
    raise Exception(f"Unsupported space type: {get_java_class(anylogic_space)}")

def __get_java_array(anylogic_model, array, jtype='Number'):
    """[INTERNAL] Convert Python array to Java array"""
    # First get class from JVM
    if jtype == 'Number':
        entry_class = anylogic_model.jvm.java.lang.Number
    elif jtype == 'double':
        entry_class = anylogic_model.jvm.double
    elif jtype == 'int':
        entry_class = anylogic_model.jvm.int
    elif jtype == 'Double':
        entry_class = anylogic_model.jvm.java.lang.Double
    elif jtype == 'Integer':
        entry_class = anylogic_model.jvm.java.lang.Integer
    # Create java array using 'py4j' given data type and array length
    jarray = anylogic_model.new_array(entry_class, len(array))
    # Populate array with values from action and cast them accordingly
    for i, v in enumerate(array):
        jarray[i] = int(v) if np.issubdtype(type(v), np.integer) else float(v)
    return jarray

def get_java_map(anylogic_model, python_dict):
    """[INTERNAL] Convert Python dictionary to Java map"""
    # Create java map using 'py4j' given data type
    jmap = anylogic_model.jvm.java.util.LinkedHashMap()
    # Populate array with values from action and cast them accordingly
    for k, v in python_dict.items():
        jmap.put(k, v)
    return jmap

def load_space(location_path):
    """[INTERNAL] Load space from given location"""
    # Load space from given location using pickle
    with open(location_path, 'rb') as f:
        space = pickle.load(f)
    return space

def save_space(space, location_path):
    """[INTERNAL] Save space to given location"""
    # First, create location directory if it does not exist
    os.makedirs(os.path.dirname(location_path), exist_ok=True)
    # Define the lock file path
    lock_file_path = location_path + '.lock'
    # Use a file lock for synchronization across processes
    with FileLock(lock_file_path):
        # Save space to given location
        with open(location_path, 'wb') as f:
            pickle.dump(space, f)
