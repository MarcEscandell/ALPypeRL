import math
from gymnasium import spaces
import numpy as np
from alpyperl import BaseAnyLogicEnv


class CartPoleEnv(BaseAnyLogicEnv):

    def __init__(self, env_config=None, disable_env_checking=True):

        # ---------------------------------------------------------------------
        # Action and observation spaces
        # ---------------------------------------------------------------------

        # Thresholds
        theta_threshold_radians = 12 * 2 * math.pi / 360.0
        x_threshold = 2.4
        # Observation space array thresholds
        high = np.array(
            [
                x_threshold * 2,
                np.finfo(np.float32).max,
                theta_threshold_radians * 2,
                np.finfo(np.float32).max
            ]
        )
        # Action/observation spaces
        self.action_space = spaces.Box(np.array([0, -1]), np.array([1, 0]), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        # Initialise AnyLogic environment experiment
        super(CartPoleEnv, self).__init__(env_config)