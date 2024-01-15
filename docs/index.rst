Welcome to **ALPypeRL** documentation!
======================================

**ALPypeRL** or *AnyLogic Python Pipe for Reinforcement Learning* is an open source library for connecting **AnyLogic** simulation models with **reinforcement learning** frameworks that are compatible with *OpenAI Gymnasium* interface (single agent).

.. important::   
   **No license** is required for single instance experimentation. **AnyLogic PLE** is **free**! Download it from `here <https://www.anylogic.com/downloads/>`_.

.. note::
   *ALPypeRL* has been developed using **ray rllib** as the base *RL framework*. *ray rllib* is an industry leading open source package for Reinforcement Learning that offers lots of interesting features. Because of that, ALPypeRL has certain dependencies to it (e.g. trained policy deployment and evaluation).

.. toctree::
   :maxdepth: 2
   :caption: General:
   
   Home
   AnyLogicConnector
   CartPoleV0
   GymSpaces
   CartPoleV1
   CartPoleV2
   CartPoleV3
   Evaluation
   RandomSeed
   DefineParamValues

.. toctree::
   :maxdepth: 1
   :caption: Advanced:

   RunFromAnyLogic
   DockerContainer
   ScaleTraining
   CustomExperiment
   CommonIssues
   API



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
