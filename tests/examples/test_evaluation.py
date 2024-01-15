import pytest
import time
import threading
import requests
from alpyperl.serve.rllib import launch_policy_server
from alpyperl import AnyLogicEnv
from ray.rllib.algorithms.ppo import PPOConfig
import platform
import subprocess
from subprocess import Popen
import os
import signal
import psutil
import logging
logger = logging.getLogger(__name__)

def launch_server(example_index, port):
    launch_policy_server(
        policy_config=PPOConfig(),
        env=AnyLogicEnv,
        trained_policy_loc=f'./tests/trained_policies/cartpole_v{example_index}',
        port=port
    )

@pytest.mark.order(2)
@pytest.mark.parametrize("example_index, eval_port", [
    (0, 3000),
    (1, 3001),
    (2, 3002),
    (3, 3003)
])
def test_run_example_evaluation(example_index, eval_port):
    logger.info("\nRunning example evaluation for cartpole_v{}".format(example_index))

    # Create a thread to launch the server
    thread = threading.Thread(target=launch_server, args=(example_index, eval_port), daemon=True)
    thread.start()

    # Wait until the server is ready
    max_retries = 60
    retries = 0
    while retries < max_retries:
        try:
            response = requests.get(f"http://localhost:{eval_port}")
            if response.status_code == 200:
                logger.info("Server is ready")
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
        retries += 1

    # If the server is still not ready after max_retries, handle it as needed
    if retries == max_retries:
        logger.error("Server did not start successfully.")
        # You may want to raise an exception or handle this case appropriately

    proj_loc = f'./resources/exported_models/cartpole_v{example_index}_eval/CartPole_v{example_index}'
    exec_file_loc = {
        'Linux': f"'{proj_loc}_linux.sh'",
        'Windows': f"'{proj_loc}_windows.bat'",
        'Darwin': f"'{proj_loc}_mac'"
    }.get(platform.system())
    command = {
        'Linux': ['bash', '-c', exec_file_loc],
        'Windows': [exec_file_loc],
        'Darwin': ['open', exec_file_loc]
    }.get(platform.system())
    # Execute model
    al_process = Popen(
        command,
        shell=(platform.system() == 'Windows'),
        start_new_session=True
    )
    # Wait for run to complete
    time.sleep(20)

    # Terminate the process
    os.killpg(os.getpgid(al_process.pid), signal.SIGTERM)

    logger.info("Test complete for example {}".format(example_index))

