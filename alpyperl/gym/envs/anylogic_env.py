import logging
import gymnasium as gym
from gymnasium import spaces
from random import randint, uniform
from alpyperl.anylogic.model.connector import AnyLogicModelConnector
import numpy as np

def create_custom_env(action_space, observation_space, lesson_configs = [], env_config: dict=None):
    """ Create a custom environment by passing an `action`, `observation` and 'lesson_configs'

    :param action_space: A valid action space: integer, double or an array of doubles
    :type action_space: gymnasium.spaces
    :param observation_space: A valid observation space as an array of doubles
    :type observation_space: gymnasium.spaces
    :param lesson_configs: A valid list of primitive types or tuples
    :type lesson_configs: list
    :param env_config: Environment configuration which includes:

        * ``'run_exported_model'``: In case you want to run an exported version 
          of the model. Otherwise it will wait for the AnyLogic model to connect. 
        * ``'exported_model_loc'``: The location of the exported model folder. 
        * ``'show_terminals'``: This only applies if running an exported model 
          and the user wants a terminal to be launched for every model instance 
          (could be useful for debugging purposes).
        * ``'server_mode_on'``: This is for internal use only. It is used to 
          flag the AnyLogic model to not be launched when serving a trained policy. 
        * ``'verbose'``: To be activated in case DEBUG logger wants to be activated. 
            
    :type env_config: dict

    :return: Returns a class definition of your custom environment with the 
        specified action and observation spaces and lesson configs
    :rtype: CustomEnv
    """
    class CustomEnv(BaseAnyLogicEnv):

        def __init__(self, env_config=None):
            # Action/observation spaces
            self.action_space = action_space
            self.observation_space = observation_space
            # Lesson configs
            self.lesson_configs = lesson_configs
            # Initialise AnyLogic environment experiment
            super(CustomEnv, self).__init__(env_config)

    return CustomEnv


class BaseAnyLogicEnv(gym.Env):
    """
    The python class that contains the AnyLogic model connection and is in 
    charge of retrieving the information required to be returned by `OpenAI 
    Gymnasium` functions such as `step` and `reset`.
    """

    metadata = {'render.modes': ['human']}

    lesson_configs = []
    currLesson = 0
    
    def validate_observation_space(self):
        """`[INTERNAL]` validates the observation space with the following conditions:
            1. Box spaces must be one dimensional. This is because ppo config is not able to accept tuple spaces with multiple spaces of different dimensions (a discrete space is considered one dimensional)
            2. A tuple space must contain more than space within it (having a tuple with one space is pointless). Necassary when parsing discrete and box spaces correctly in evaluation
            3. Should not have a tuple within another tuple. This would be pointless to do and would also make it much more complex to convert for training and evaluation
        """
        if isinstance(self.observation_space, spaces.Box):
            assert len(self.observation_space.shape) == 1, "Box observation space must be one dimensional"
        elif isinstance(self.observation_space, spaces.Tuple):
            assert len(self.observation_space) > 1 , "Tuple observation space should contain more than one element"
            for i in range(len(self.observation_space)):
                if isinstance(self.observation_space[i], spaces.Box):
                    assert(len(self.observation_space[i].shape) == 1), "Box observation space within tuple must be one dimensional"
                elif isinstance(self.observation_space[i], spaces.Tuple):
                    raise Exception("Observation space should not have a tuple within a tuple")

            
    def sample_tuple(self, config_tuple):
        """`[INTERNAL]` Samples a tuple variable from within the current lesson_config
            Tuple must be specified in the form (value1, value2, type) => where type is either int or float. The method returns a randomly sampled value between value1 and value2
            Note: value1 must be lower than value2
        """
        if len(config_tuple) == 3:
            if config_tuple[2] == int:
                if isinstance(config_tuple[0], int) and isinstance(config_tuple[1], int):
                    assert config_tuple[0] < config_tuple[1], ("invalid tuple ", config_tuple)
                    return randint(config_tuple[0], config_tuple[1])
                else:
                    raise Exception("invalid tuple ", config_tuple)
            elif config_tuple[2] == float:
                if (isinstance(config_tuple[0], int) or isinstance(config_tuple[0], float)) and (isinstance(config_tuple[1], int) or isinstance(config_tuple[1], float)):
                    assert config_tuple[0] < config_tuple[1], ("invalid tuple ", config_tuple)
                    return uniform(config_tuple[0], config_tuple[1])
                else:
                    raise Exception("invalid tuple ", config_tuple)
            else:
              raise Exception("invalid type paramater ", config_tuple[2])  
        else:
            raise Exception("invalid tuple structure ", config_tuple)


    def sample_config(self):
        """`[INTERNAL]` Samples the values for the lesson config. If the list is empty then there is no config so None is returned
            Iterates through each entry in the list. If it is a primitive type int, float, str or bool then it is added to the config array 
            If the entry is a tuple then a value is sampled and added to the array
        """
        if not isinstance(self.lesson_configs, list):
            raise Exception("config space needs to be a list");
        elif len(self.lesson_configs) == 0:
            return None

        configs = self.lesson_configs[self.currLesson]

        object_class = self.anylogic_model.jvm.Object
        object_array = self.anylogic_model.new_array(object_class, len(configs))

        for i, v in enumerate(configs):
            if isinstance(v, int) or isinstance(v, float) or isinstance(v, str) or isinstance(v, bool):
                object_array[i] = v
            elif isinstance(v, tuple):
                object_array[i] = self.sample_tuple(v)
            else:
                raise Exception("invalid lesson config variable ", v)
                
        return object_array

    """ 
    Parses the observation space into the correct format for the policy to accept depending on the space
    Discrete => return first entry as primitive type
    Box => return first entry as list
    Discrete => return full observation as list
    """
    def parse_observation(self, observation):
        if isinstance(self.observation_space, spaces.Discrete):
            return observation[0]
        elif isinstance(self.observation_space, spaces.Box):
            return list(observation[0])
        return list(observation)


    def __init__(
        self,
        env_config: dict = {
            'run_exported_model': True,
            'exported_model_loc': './exported_model',
            'show_terminals': False,
            'server_mode_on': False,
            'verbose': False
        },
        disable_env_checking: bool = True
    ):
        """
        Internal AnyLogic environment wrapper constructor

        :param env_config: Environment configuration which includes:

            * ``'run_exported_model'``: In case you want to run an exported 
              version of the model. Otherwise it will wait for the AnyLogic 
              model to connect. 
            * ``'exported_model_loc'``: The location of the exported model folder. 
            * ``'show_terminals'``: This only applies if running an exported 
              model and the user wants a terminal to be launched for every 
              model instance (could be useful for debugging purposes). 
            * ``'server_mode_on'``: This is for internal use only. It is used 
              to flag the AnyLogic model to not be launched when serving a 
              trained policy. 
            * ``'verbose'``: To be activated in case DEBUG logger wants to be 
              activated. 

        :type env_config: dict
        
        """

        #validate observation space is in correct format
        self.validate_observation_space()

        # Initialise `env_config` to avoid problems when handling `None`
        self.env_config = env_config if env_config is not None else []

        # Initialise logger
        verbose = (
            'verbose' in self.env_config
            and self.env_config['verbose']
        )
        # Only log message from `alpyperl`
        ch = logging.StreamHandler()
        ch.addFilter(logging.Filter('alpyperl'))
        # Create logger configuration
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format=f"%(asctime)s [%(name)s][%(levelname)8s] %(message)s",
            handlers=[ch],
        )
        self.logger = logging.getLogger(__name__)

        # Check if server mode is on.
        # When loading a trained policy, there's no need to launch the model
        # as it causes an overhead.
        # Yet, obervation space is required to be returned. In that case,
        # it is only necessary to return a sample.
        self.server_mode_on = (
            'server_mode_on' in self.env_config
            and self.env_config['server_mode_on']
        )

        # Launch or connect to AnyLogic model using the connector and launcher.
        if not self.server_mode_on:
            self.anylogic_connector = AnyLogicModelConnector(
                run_exported_model=(
                    self.env_config['run_exported_model'] 
                    if 'run_exported_model' in self.env_config 
                    else True
                ),
                exported_model_loc=(
                    self.env_config['exported_model_loc'] 
                    if 'exported_model_loc' in self.env_config 
                    else './exported_model'
                ),
                show_terminals=(
                    self.env_config['show_terminals'] 
                    if 'show_terminals' in self.env_config 
                    else False
                )
            )
            # The gateway is the direct interface to the AnyLogic model.
            self.anylogic_model = self.anylogic_connector.gateway

            # Initialise and prepare the model by calling `reset` method.
            self.anylogic_model.reset(self.sample_config())
            
            self.logger.info("AnyLogic model has been initialized correctly!")

    def step(self, action):
        """`[INTERNAL]` Basic function for performing 'steps' in order for the simulation to
        move on. It requires an `action` as an input. This action can be of
        different types (including an array of values).
        """

        # Run fast simulation until next action is required (which will be
        # controlled and requested from the AnyLogic model)
        if not self.server_mode_on:
            # Parse action to a type that can be consumed by AnyLogic model.
            action_parsed = self.__parse_action(action)
            # Create a java object that can handle multiple java types
            # `ActionSpace` is a class that has been defined in AnyLogic from
            # the ALPypeRLConnector.
            action_space = self.anylogic_model.jvm.com.alpyperl.ActionSpace(action_parsed)
            # Pass action to AnyLogic model.
            self.anylogic_model.step(action_space)

        # Get observation state or sample if in server mode.
        state = (
            np.asarray(self.parse_observation(self.anylogic_model.getState()))
            if not self.server_mode_on
            else self.observation_space.sample()
        )

        # Get 'current' reward (not cumulated) or dummy 0 if in server mode
        # It is assumed that reward will always be an scalar.
        reward = (
            self.anylogic_model.getReward()
            if not self.server_mode_on
            else 0
        )

        # Check if simulation has finished.
        # Simulation length can be fixed or subject to other
        # conditions (e.g. system fails earlier and continuation is non-sense)
        done = (
            self.anylogic_model.hasFinished()
            if not self.server_mode_on
            else True
        )

        # Return tuple: STATE, REWARD, DONE, INFO
        return state, reward, done, False, {}


    def reset(self, *, seed=None, options=None):
        """`[INTERNAL]` Reset function will restart the AnyLogic model to its initial status
        and return the new initial state"""
        # Reset simulation to restart from initial conditions
        new_state = (
            # Samples and passes lesson config to set up initial state
            np.asarray(self.parse_observation(self.anylogic_model.reset(self.sample_config())))
            if not self.server_mode_on
            else self.observation_space.sample()
        )
        # Return tuble: STATE, INFO
        return new_state, {}


    def render(self):
        """`[INTERNAL]` Whether any visualisation will be displayed or not, depends on the
        user when decides to export an experiment with visualisation or not"""
        pass

    def close(self):
        """`[INTERNAL]` Close executables if any was created"""
        self.anylogic_connector.close_connection()

    def parse_tuple(self, tuple_action, tuple_space):
        """`[INTERNAL]` Takes in a tuple action and converts into a format that can be accepted by the anylogic model
        Iterates through each space in the tuple and value in the action. If it is a discrete space then convert value to int and add to object array
        If box then convert each element to a float, add to double array and then add the double array to the object array
        """
        object_class = self.anylogic_model.jvm.Object
        object_array = self.anylogic_model.new_array(object_class, len(tuple_action))
        for i, v in enumerate(tuple_action):
            if isinstance(tuple_space[i], spaces.Discrete):
                object_array[i] = int(v)
            elif isinstance(tuple_space[i], spaces.Box):
                assert len(tuple_space[i].shape) == 1, "box action spaces must be 1 dimensional"
                double_class = self.anylogic_model.jvm.double
                double_array = self.anylogic_model.new_array(double_class, v.size)
                for j, w in enumerate(v):
                    double_array[j] = float(w)
                object_array[i] = double_array

            # Prevent nested tuples as they serve no real purpose and only make it more complicated to convert
            elif isinstance(tuple_space[i], spaces.Tuple):
                raise Exception("action space cannot have a nested tuple")
        return object_array

    def __parse_action(self, action):
        """`[INTERNAL]`Parse the action from `numpy` to a primitive type that can be taken by
        java"""
        if isinstance(self.action_space, spaces.Discrete):
            return int(action)
        elif isinstance(self.action_space, spaces.Tuple):
            object_array = self.parse_tuple(action, self.action_space)
            return object_array
        elif isinstance(self.action_space, spaces.Box) and action.size == 1:
            return float(action[0])

        assert len(self.action_space.shape) == 1, "Box action space should be 1 dimensional"
        # Assume it is an array and create a double[] type that can be consumed
        # by java model
        # First get double class from JVM
        double_class = self.anylogic_model.jvm.double
        # Create double array using 'py4j'
        double_array = self.anylogic_model.new_array(double_class, action.size)
        # Populate array with values from action
        for i, v in enumerate(action):
            double_array[i] = float(v)
        return double_array

    def next_lesson(self):
        """`[INTERNAL]`Called from within the training script. Increments the current lesson count so that the environment 
        uses the next set of configs when setting up the model"""
        self.currLesson += 1
        if len(self.lesson_configs) <= self.currLesson:
            raise Exception(str(self.currLesson + 1) + " exceeds lesson count " + str(len(self.lesson_configs)))
