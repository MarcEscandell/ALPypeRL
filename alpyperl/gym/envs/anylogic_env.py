import logging
import gymnasium as gym
from gymnasium import spaces
from alpyperl.anylogic.model.connector import AnyLogicModelConnector
import numpy as np

def create_custom_env(action_space, observation_space, env_config: dict=None):
    """ Create a custom environment by passing an `action` and `observation`

    :param action_space: A valid action space: integer, double or an array of doubles
    :type action_space: gymnasium.spaces
    :param observation_space: A valid observation space as an array of doubles
    :type observation_space: gymnasium.spaces.Box
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
        specified action and observation spaces
    :rtype: CustomEnv
    """
    class CustomEnv(BaseAnyLogicEnv):

        def __init__(self, env_config=None):
            # Action/observation spaces
            self.action_space = action_space
            self.observation_space = observation_space
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
            self.anylogic_model.reset()
            
            self.logger.info("AnyLogic model has been initialized correctly!")

    def step(self, action):
        """`[INTERNAL]` Basic function for performing 'steps' in order for the simulation to
        move on. It requires an `action` as an input. This action can be of
        different types (including an array of values).
        """

        # Get observation state or sample if in server mode.
        state = (
            np.asarray(list(self.anylogic_model.getState()))
            if not self.server_mode_on
            else self.observation_space.sample()
        )

        # Run fast simulation until next action is required (which will be
        # controlled and requested from the AnyLogic model)
        if not self.server_mode_on:
            # Parse action to a type that can be consumed by AnyLogic model.
            action_parsed = self.__parse_action(action)
            # Create a java object that can handle multiple java types
            # `ActionSpace` is a class that has been defined in AnyLogic from
            # the ALPypeRLConnector.
            action_space = self.anylogic_model.jvm.com.alpype.ActionSpace(action_parsed)
            # Pass action to AnyLogic model.
            self.anylogic_model.step(action_space)

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
            np.asarray(list(self.anylogic_model.reset()))
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

    def __parse_action(self, action):
        """Parse the action from `numpy` to a primitive type that can be taken by
        java"""
        if isinstance(self.action_space, spaces.Discrete):
            return int(action)
        elif isinstance(self.action_space, spaces.Box) and action.size == 1:
            return float(action[0])

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
