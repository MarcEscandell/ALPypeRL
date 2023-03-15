###########################################
Running ``alpyperl`` in a docker container
###########################################

If you are planning to develop your project in a **docker container**, you must be aware of the following:

* Connection between **python** and **AnyLogic** will potentially fail. This is due to *alpyperl* trying to connect to a port that is not mapped between the **docker container** and **docker host**.
* Howerver, this is possible when running with an **exported model** as execution of the model will take place in the container. In the terminal, AnyLogic will share the port in case you want to access the model UI.

    .. code-block:: console
        :emphasize-lines: 6

        root@e71cad1ec378:/workspaces/ALPypeRL# python alpyperl/examples/cartpole_v0/train.py 
        2023-03-14 22:10:20,619 INFO worker.py:1553 -- Started a local Ray instance.
        (RolloutWorker pid=3831) /usr/local/lib/python3.10/site-packages/gymnasium/spaces/box.py:127: UserWarning: WARN: Box bound precision lowered by casting to float32
        (RolloutWorker pid=3831)   logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
        (RolloutWorker pid=3831) chmod: cannot access 'chromium/chromium-linux64/chrome': No such file or directory
        (RolloutWorker pid=3831) Couldn't find browser: chromium/chromium-linux64/chrome. Attempting to open system-default browser for url: http://localhost:23109
        (RolloutWorker pid=3831) [INFO] ALPypeRLConnector - Welcome to the 'ALPypeRL Connector'!

  If you are planning to host your **trained policy** in a *docker container* you should apply the same logic: you must map the port to the host machine so it is accessible. Then, just remember to set the right port in the ``ALPypeRLConnector``.