import uvicorn
from typing import List
from fastapi import FastAPI, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import numpy as np
from alpyperl.gym.envs import utils
from alpyperl import BaseAnyLogicEnv
from gymnasium import spaces
from gymnasium.spaces.utils import unflatten, flatten
import os


def launch_policy_server(
    policy_config,
    env,
    env_config=None,
    trained_policy_loc='./"trained_policy"',
    host="0.0.0.0",
    port=3000
):
    """Launch server and host trained policy to allow requests. The server
    requires an observation in the form of an array and will return and action
    (type depends on action space)

    :param policy_config: It refers to the policy (also refered as *RL algorithm*) that 
        will be trained. It must be an instance of **rllib algorithms** (check here 
        for more `information <https://docs.ray.io/en/latest/rllib/rllib-algorithms.html>`_)
    :type env_config: ray.rllib.algorithms
    :param env: The environment or **AnyLogic simulation** that the policy will use
        to train against.
    :type env: alpyperl.BaseAnyLogicEnv
    :param env_config: Any option that will be consumed by the environment. Options
        available are: *TODO:*
    :type env_config: dict
    :param trained_policy_loc: The location of the **rllib** trained policy. Be mindful
        that the rllib policy creates an aditional folder when calling ``save``.
        ``checkpoint_dir = policy.save("./resources/trained_policies/cartpole_v0")``
    :type trained_policy_loc: str
    :param host: The host ID to be used. Defaults to ``localhost`` or ``0.0.0.0``
    :type host: str
    :param port: The port the service will connect to. Defaults to ``3000``
    :type port: int
    """

    # Set server flag on to avoid loading the AnyLogic model
    if env_config is None:
        env_config = {}
    env_config['server_mode_on'] = True
    # Default checkpoint directory to 'trained_policy_loc' if not provided
    if 'checkpoint_dir' not in env_config:
        env_config['checkpoint_dir'] = trained_policy_loc

    # Re-create policy configuration with no workers and avoid launching
    # unnecessary models
    policy = (
        policy_config
            .rollouts(num_rollout_workers=0)
            .environment(env=env, env_config=env_config)
            .build()
    )

    # Restore policy state from given checkpoint
    policy.restore(trained_policy_loc)

    # Load observation space, so observation received from server can be parsed
    # to the correct format
    observation_space = utils.load_space(
        f"{trained_policy_loc}/alpyperl_spaces/observation_space.pkl"
    )
    action_space = utils.load_space(
        f"{trained_policy_loc}/alpyperl_spaces/action_space.pkl"
    )

    # Initialise FastAPI application server
    app = FastAPI()

    @app.get("/")
    async def greetings():
        html_content = """
            <p style=\"text-align:center;font-size:35px\">
                Welcome to <b>ALPypeRL</b> trained policy serving!
            </p>
            <p style=\"text-align:center;font-size:15px\">
                You have accessed the API server. Append <b>'/docs'</b> at the 
                end of this url to access the documentation on the methods available.
            </p>
        """
        return HTMLResponse(content=html_content, status_code=200)

    @app.post("/predict")
    async def predict_next_action(observation: List[float]):
        # Check documentation at https://docs.ray.io/en/latest/serve/tutorials/rllib.html
        action = policy.compute_single_action(
            observation=unflatten(observation_space, np.asarray(observation)),
            explore=False
        )
        # Format response
        response = {
            "observation": observation,
            "action": flatten(action_space, action).tolist()
        }

        return JSONResponse(content=jsonable_encoder(response), status_code=200)

    @app.get("/get_trained_policy_loc")
    async def get_trained_policy_loc():
        # Format response
        response = {
            "trained_policy_loc": os.path.abspath(trained_policy_loc)
        }

        return JSONResponse(content=jsonable_encoder(response), status_code=200)

    # Launch server
    uvicorn.run(app, host=host, port=port)