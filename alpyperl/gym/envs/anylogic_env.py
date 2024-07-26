import logging
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces.utils import unflatten, flatten
from alpyperl.anylogic.model.connector import AnyLogicModelConnector
import numpy as np
from alpyperl.gym.envs import utils
import os
import time


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
        * ``'checkpoint_dir'``: The location of the checkpoint directory to 
          store the action and observation spaces in case they are defined
          in the AnyLogic model. This is required mainly during policy
          evaluation.

            
    :type env_config: dict

    :return: Returns a class definition of your custom environment with the 
        specified action and observation spaces
    :rtype: CustomEnv
    """
    class CustomEnv(BaseAnyLogicEnv):

        def __init__(self, env_config=None):
            # Action/observation spaces.
            self.action_space = action_space
            self.observation_space = observation_space
            # Initialise AnyLogic environment experiment.
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
            'verbose': False,
            'checkpoint_dir': './trained_policies',
            'env_params': {}
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
            * ``'checkpoint_dir'``: The location of the checkpoint directory to 
              store the action and observation spaces in case they are defined
              in the AnyLogic model. This is required mainly during policy
              evaluation.
            * ``'env_params'``: The environment custom parameter values (e.g., 
              ``cartpole_mass``) as a dictionary

        :type env_config: dict
        
        """
        # Initialise `env_config` to avoid problems when handling `None`.
        self.env_config = env_config if env_config is not None else []

        # Initialise logger.
        verbose = (
            'verbose' in self.env_config
            and self.env_config['verbose']
        )
        # Only log message from `alpyperl`.
        ch = logging.StreamHandler()
        ch.addFilter(logging.Filter('alpyperl'))
        # Create logger configuration.
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
        # Check if 'Action' and 'Observation' spaces have been defined
        # This value will be false if spaces have been defined in the AnyLogic.
        self.spaces_exist = (
            (hasattr(self, 'action_space') and self.action_space is not None)
            or (hasattr(self, 'observation_space') and self.observation_space is not None)
        )
        # Initialize checkpoint dir to be used to store spaces.
        self.checkpoint_dir = (
            self.env_config['checkpoint_dir']
            if 'checkpoint_dir' in self.env_config
            else './trained_policies'
        )
        # Initialize custom parameter values.
        self.env_params = (
            self.env_config['env_params']
            if 'env_params' in self.env_config
            else {}
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

            # Initialise and prepare the model by calling `init()` method.
            self.anylogic_model.init()

            # Check if spaces have been defined from AnyLogic model.
            if self.anylogic_model.hasSpacesDefined():

                # Before setting the spaces, make sure that spaces have not
                # already been defined by inheritance of 'BaseAnyLogicEnv'.
                if self.spaces_exist:
                    raise Exception(
                        "Action/observation spaces have already been defined! "
                        "Please ensure that they are only defined in one place. "
                        "You must choose either from your AnyLogic model or "
                        "from your custom environment in python."
                    )

                self.logger.debug("Spaces have been defined in AnyLogic model")
                # Get action space from AnyLogic model.
                self.anylogic_action_space = self.anylogic_model.getActionSpace()
                # Parse action space from AnyLogic model to gym.spaces.
                self.action_space = utils.parse_anylogic_rl_space(
                    anylogic_model=self.anylogic_model,
                    anylogic_rl_space=self.anylogic_action_space
                )
                # Get observation space from AnyLogic model.
                self.anylogic_observation_space = self.anylogic_model.getObservationSpace()
                # Parse observation space from AnyLogic model to gym.spaces
                self.observation_space = utils.parse_anylogic_rl_space(
                    anylogic_model=self.anylogic_model,
                    anylogic_rl_space=self.anylogic_observation_space
                )
            elif not self.spaces_exist:
                raise Exception(
                    "Action/observation spaces have not been defined! "
                    "Please ensure that they are defined either in your "
                    "AnyLogic model or in your custom environment in python."
                )

            self.logger.info("AnyLogic model has been initialized correctly!")

        elif self.server_mode_on and not self.spaces_exist:
            # When serving a trained policy, it is necessary to have the
            # observation space defined. Otherwise, it will not be possible
            # to unflatten the observation sample.
            # This will be the case when the policy was trained using spaces
            # defined in the AnyLogic model.
            self.observation_space = utils.load_space(
                f"{self.checkpoint_dir}/alpyperl_spaces/observation_space.pkl"
            )
            self.action_space = utils.load_space(
                f"{self.checkpoint_dir}/alpyperl_spaces/action_space.pkl"
            )


    def step(self, action):
        """`[INTERNAL]` Basic function for performing 'steps' in order for the simulation to
        move on. It requires an `action` as an input. This action can be of
        different types (including an array of values).
        """
        # Check if AnyLogic 'ObservationSpace' has been parsed. This is necessary
        # so observation can be flattened in the AnyLogic side.
        if (
            not self.server_mode_on 
            and (
                not hasattr(self, 'anylogic_observation_space') 
                or self.anylogic_observation_space is None
            )
        ):
            self.anylogic_observation_space = utils.parse_gym_to_anylogic_rl_space(
                anylogic_model=self.anylogic_model,
                observation_space=self.observation_space
            )
        # Run fast simulation until next action is required (which will be
        # controlled and requested from the AnyLogic model).
        if not self.server_mode_on:
            # Flatten action
            action_parsed = flatten(self.action_space, action)
            # Check if AnyLogic 'ActionSpace' has been parsed. This is necessary
            # so action can be unflattened in the AnyLogic side.
            if not hasattr(self, 'anylogic_action_space') or self.anylogic_action_space is None:
                self.anylogic_action_space = utils.parse_gym_to_anylogic_rl_space(
                    anylogic_model=self.anylogic_model,
                    action_space=self.action_space
                )
            # Convert flatten action to AnyLogic 'RLAction' together with 
            # AnyLogic 'ActionSpace' so unfaltten operation can be performed.
            action_space = utils.get_anylogic_rl_action(
                anylogic_model=self.anylogic_model,
                flattened_action=action_parsed,
                anylogic_action_space=self.anylogic_action_space
            )
            # Pass action to AnyLogic model.
            self.anylogic_model.step(action_space)
            
        # Get observation state or sample if in server mode.
        state = (
            unflatten(
                self.observation_space,
                np.asanyarray(self.anylogic_model.getState(self.anylogic_observation_space))
            )
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
        if not self.server_mode_on:
            # Initialize seed by retrieving it from AnyLogic model
            if seed is None:
                seed = self.anylogic_model.getSeed()
            else:
                raise Exception("Passing a custom seed is not supported!")
            # We need the following line to seed self.np_random
            super().reset(seed=seed)
        # Check if AnyLogic 'ObservationSpace' has been parsed. This is necessary
        # so observation can be flattened in the AnyLogic side.
        if (
            not self.server_mode_on 
            and (
                not hasattr(self, 'anylogic_observation_space') 
                or self.anylogic_observation_space is None
            )
        ):
            self.anylogic_observation_space = utils.parse_gym_to_anylogic_rl_space(
                anylogic_model=self.anylogic_model,
                observation_space=self.observation_space
            )
        # Reset simulation to restart from initial conditions.
        new_state = (
            unflatten(
                self.observation_space,
                np.asanyarray(self.anylogic_model.reset(
                    self.anylogic_observation_space,
                    utils.get_java_map(self.anylogic_model, self.env_params)
                ))
            )
            if not self.server_mode_on
            else self.observation_space.sample()
        )
        # Save alpyperl spaces to a file if they have not been saved yet.
        self.__save_spaces_if_missing()
        # Return tuble: STATE, INFO.
        return new_state, {}


    def render(self):
        """`[INTERNAL]` Whether any visualisation will be displayed or not, depends on the
        user when decides to export an experiment with visualisation or not"""
        pass

    def close(self):
        """`[INTERNAL]` Close executables if any was created"""
        self.__save_spaces_if_missing()
        self.anylogic_connector.close_connection()

    def __save_spaces_if_missing(self):
        """`[INTERNAL]` Save ALPypeRL spaces to a file"""
        # Save observation and action space if it has been defined in the
        # AnyLogic model.
        # NOTE: Since there could be multiple AnyLogic models running at the
        # same time, it is necessary to create a folder first so the other instances
        # do not overwrite the file.
        if not self.server_mode_on and not os.path.exists(f"{self.checkpoint_dir}/alpyperl_spaces/"):
            utils.save_space(self.observation_space, f"{self.checkpoint_dir}/alpyperl_spaces/observation_space.pkl")
            utils.save_space(self.action_space, f"{self.checkpoint_dir}/alpyperl_spaces/action_space.pkl")
            self.anylogic_model.jvm.com.alpype.RLSpace.save(
                self.anylogic_observation_space,
                os.path.abspath(f"{self.checkpoint_dir}/alpyperl_spaces/observation_space.ser")
            )
            self.anylogic_model.jvm.com.alpype.RLSpace.save(
                self.anylogic_action_space,
                os.path.abspath(f"{self.checkpoint_dir}/alpyperl_spaces/action_space.ser")
            )
            self.logger.info(f"ALPypeRL spaces have been saved successfully at '{self.checkpoint_dir}'")
