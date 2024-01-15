import pytest
from alpyperl.gym.envs import utils
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim, flatten, unflatten
import numpy as np
from alpyperl import AnyLogicEnv


@pytest.fixture(scope="module")
def anylogic_model():
    anylogic_model = AnyLogicEnv(
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './resources/exported_models/cartpole_v0'
        }
    )
    yield anylogic_model.anylogic_model
    anylogic_model.close()

@pytest.mark.parametrize("gym_space", [
    (spaces.Discrete(n=2)),
    (spaces.Discrete(n=12, start=123))
])
def test_convert_gym_discrete_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_discr_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check that space size matches
    assert anylogic_discr_space.spaceSize() == 1
    # Check that start value matches
    assert anylogic_discr_space.getStartValue() == gym_space.start
    # Check sample size matches
    assert anylogic_discr_space.sampleSize() == gym_space.n

@pytest.mark.parametrize("gym_space", [
    (spaces.Box(low=0.0, high=1.0)),
    (spaces.Box(low=np.array([0.0, 1.0]), high=np.array([1.2, 2.3]))),
    (spaces.Box(low=1.5, high=2.9, shape=(4,))),
    # NOTE: Not supported yet
    #(spaces.Box(low=np.array([[0.0, 1.0], [1.2, 2.3]]), high=np.array([[1.0, 2.0], [2.2, 3.3]])))
])
def test_convert_gym_box_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_box_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check that size matches
    assert anylogic_box_space.sampleSize() == flatdim(gym_space)
    # Check tht space size matches
    assert anylogic_box_space.spaceSize() == flatdim(gym_space)
    # Check spaces boundaries
    if flatdim(gym_space) == 1:
        assert anylogic_box_space.lb() == gym_space.low
        assert anylogic_box_space.ub() == gym_space.high
    elif flatdim(gym_space) > 1 and len(gym_space.shape) == 1:
        for col in range(gym_space.shape[0]):
            assert anylogic_box_space.lb(col) == gym_space.low[col]
            assert anylogic_box_space.ub(col) == gym_space.high[col]
    else:
        for row in range(gym_space.shape[0]):
            for col in range(gym_space.shape[1]):
                assert anylogic_box_space.lb(row, col) == gym_space.low[row, col]
                assert anylogic_box_space.ub(row, col) == gym_space.high[row, col]

@pytest.mark.parametrize("gym_space", [
    (spaces.MultiBinary(n=2)),
    (spaces.MultiBinary(n=123))
])
def test_convert_gym_multibinary_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_mbinary_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check that size matches
    assert anylogic_mbinary_space.sampleSize() == gym_space.n
    # Check that space size matches
    assert anylogic_mbinary_space.spaceSize() == gym_space.n

@pytest.mark.parametrize("gym_space", [
    (spaces.MultiDiscrete(nvec=[2, 3])),
    (spaces.MultiDiscrete(nvec=[2, 3, 4, 5, 6], start=[10, 20, 30, 40, 5]))
])
def test_convert_gym_multidiscrete_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_mdiscr_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check that size matches
    assert anylogic_mdiscr_space.spaceSize() == len(gym_space.nvec)
    assert anylogic_mdiscr_space.sampleSize() == sum(gym_space.nvec)
    # Validate start values
    for i in range(len(gym_space.nvec)):
        assert anylogic_mdiscr_space.getStartValue(i) == gym_space.start[i]
    # Validate nvec values
    for i in range(len(gym_space.nvec)):
        assert anylogic_mdiscr_space.get(i) == gym_space.nvec[i]

@pytest.mark.parametrize("gym_space", [
    (spaces.Tuple(spaces=(spaces.Discrete(n=2), spaces.Discrete(n=3)))),
    (spaces.Tuple(
        spaces=[
            spaces.Discrete(n=5, start=2), 
            spaces.Box(low=0.0, high=1.32, shape=(3, 4))
        ]
    )),
    (spaces.Tuple(spaces=[
        spaces.Discrete(n=3),
        spaces.Tuple(spaces=[
            spaces.Box(low=-1, high=100),
            spaces.MultiDiscrete(nvec=[2, 3], start=[-2, 3])
        ])
    ]))
])
def test_convert_gym_tuple_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_tuple_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check space sizes
    assert anylogic_tuple_space.getNumSpaces() == len(gym_space.spaces)
    # Check sample size
    assert anylogic_tuple_space.sampleSize() == flatdim(gym_space)

@pytest.mark.parametrize("gym_space", [
    (spaces.Dict(spaces={
        "space-a": spaces.Discrete(n=2), "space-b": spaces.Discrete(n=3)
    })),
    (spaces.Dict(spaces={
        "space-a": spaces.Discrete(n=2), 
        "space-b": spaces.Dict(spaces={
            "space-c": spaces.Box(low=np.array([0.1, 0.2]), high=np.array([0.3, 0.4])),
            "space-d": spaces.MultiBinary(n=2)
        })
    }))
])
def test_convert_gym_dict_space_to_anylogic(anylogic_model, gym_space):
    # Test if space can be converted and no errors are thrown
    anylogic_dict_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Check space sizes
    assert anylogic_dict_space.getNumSpaces() == len(gym_space.spaces)
    # Check sample size
    assert anylogic_dict_space.sampleSize() == flatdim(gym_space)
    # Check space names/keys
    for gk, alk in zip(gym_space.spaces.keys(), list(anylogic_dict_space.getSpaceNames())):
        assert gk == alk

def flatten_space_sample(space_sample):
    def flatten_recursive(data):
        if isinstance(data, tuple):
            flattened = []
            for item in data:
                flattened.extend(flatten_recursive(item))
            return flattened
        elif isinstance(data, dict):
            flattened_values = []
            for key, value in data.items():
                flattened_value = flatten_recursive(value)
                flattened_values.extend(flattened_value)
            return flattened_values
        else:
            return [data]

    flattened_data = flatten_recursive(space_sample)
    return np.concatenate([np.ravel(item) for item in flattened_data])

@pytest.mark.parametrize("gym_space, space_type", [
    (spaces.Box(low=0.0, high=1.0), "action_space"),
    (spaces.Discrete(n=2), "action_space"),
    (spaces.Discrete(n=12, start=123), "observation_space"),
    (spaces.MultiDiscrete(nvec=[2, 5]), "observation_space"),
    (spaces.MultiDiscrete(nvec=[7, 8, 2], start=[99, -236, 0]), "observation_space"),
    (spaces.Box(low=np.array([0.0, 12, -2.3]), high=np.array([1.0, 200, 0])), "action_space"),
    (spaces.MultiBinary(n=6), "observation_space"),
    (spaces.Tuple(spaces=[
        spaces.Discrete(n=2, start=-5), spaces.MultiDiscrete(nvec=[7, 2, 1])
    ]), "observation_space"),
    (spaces.Dict(spaces={
        "space-a": spaces.Box(low=-9.3, high=147.0),
        "space-b": spaces.Tuple(spaces=[
            spaces.MultiBinary(n=3), 
            spaces.MultiDiscrete(nvec=[1, 2, 3], start=[9, 5, -9])
        ])
    }), "action_space")
])
def test_get_anylogic_rl_space(anylogic_model, gym_space, space_type):
    # Test if space can be converted and no errors are thrown
    anylogic_space = utils.__convert_gym_space_to_anylogic(anylogic_model, gym_space)
    # Create AnyLogic space
    if space_type == "action_space":
        anylogic_space = utils.parse_gym_to_anylogic_rl_space(
            anylogic_model=anylogic_model,
            action_space=gym_space
        )
    elif space_type == "observation_space":
        anylogic_space = utils.parse_gym_to_anylogic_rl_space(
            anylogic_model=anylogic_model,
            observation_space=gym_space
        )
    # Create a sample of the space
    space_sample = gym_space.sample()
    # Flatten sample in gym and AnyLogic and compare
    gym_flatten =  flatten(gym_space, space_sample)
    anylogic_flatten = anylogic_model.jvm.com.alpype.RLSpace.flatten(
        anylogic_space,
        utils.__get_java_array(
            anylogic_model,
            flatten_space_sample(space_sample) if not np.isscalar(space_sample) else [space_sample],
            jtype='Number'
        )
    )
    assert gym_flatten.tolist() == list(anylogic_flatten)

@pytest.mark.parametrize("gym_space", [
    (spaces.Discrete(n=2)),
    (spaces.Discrete(n=12, start=123)),
    (spaces.Box(low=-1.23, high=321)),
    (spaces.MultiBinary(n=2000)),
    (spaces.MultiDiscrete(nvec=[2, 5, 10], start=[-70, 1, 300])),
    (spaces.Tuple(spaces=[
        spaces.MultiDiscrete(nvec=[2, 10]),
        spaces.Box(low=np.array([0.1, 0.1]), high=np.array([1.0, 1.0]))
    ]))
])
def test_get_anylogic_rl_action(anylogic_model, gym_space):
    # Sample space
    action_sample = gym_space.sample()
    # Flatten action
    flatten_action = flatten(gym_space, action_sample)
    # Create AnyLogic sample
    anylogic_action = utils.get_anylogic_rl_action(
        anylogic_model=anylogic_model,
        flattened_action=flatten_action,
        anylogic_action_space=utils.parse_gym_to_anylogic_rl_space(
            anylogic_model=anylogic_model,
            action_space=gym_space
        )
    )
    assert flatten_space_sample(action_sample).tolist() == list(anylogic_action.getActions())
    # Validate individual actions
    for i, action in enumerate(flatten_space_sample(action_sample)):
        if np.issubdtype(type(action), np.integer):
            assert action == anylogic_action.getInt(i)
        # NOTE: It seems for 'spaces.Tuple' and 'spaces.Dict', discrete values 
        # are parsed to float
        elif (
            np.issubdtype(type(action), np.floating)
            and not isinstance(gym_space, (spaces.Tuple, spaces.Dict))
        ):
            assert action == anylogic_action.getDouble(i)

    