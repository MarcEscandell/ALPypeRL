import gymnasium as gym
from gymnasium import spaces
from alpyperl.anylogic.model.connector import AnyLogicModelConnector
import numpy as np


class AnyLogicEnv(gym.Env):
    """The python class that contains the AnyLogic model conection and is in 
    charge of retrieving the information required to be returned by OpenAI 
    Gymnasium functions such as `step` and `reset`.
    """

    metadata = {'render.modes': ['human']}

    def __init__(
        self,
        env_config={
            'run_exported_model': True,
            'exported_model_loc': './exported_model',
            'show_terminals': False,
            'server_mode_on': False
        },
        disable_env_checking=True
    ):
        # Initialise `env_config` to avoid problems when handling `None`
        self.env_config = env_config if env_config is not None else []
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
                ),
            )
            # The gateway is the direct interface to the AnyLogic model.
            self.anylogic_model = self.anylogic_connector.gateway

            # Initialise and prepare the model by calling `reset` method.
            self.anylogic_model.reset()

    def step(self, action):
        """Basic function for performing 'steps' in order for the simulation to
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
            np.asarray(list(self.anylogic_model.getState()))
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
        """Reset function will restart the AnyLogic model to its initial status
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
        """Whether any visualisation will be displayed or not, depends on the
        user when decides to export an experiment with visualisation or not"""
        pass

    def close(self):
        """Close executables if any was created"""
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
