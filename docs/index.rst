Welcome to **ALPypeRL** documentation!
======================================

**ALPypeRL** or *AnyLogic Python Pipe for Reinforcement Learning* is an open source library for connecting **AnyLogic** simulation models with **reinforcement learning** frameworks that are compatible with *OpenAI Gymnasium* interface (single agent).

.. note::
   *ALPypeRL* has been developed using **ray rllib** as the base *RL framework*. *ray rllib* is an industry leading open source package for Reinforcement Learning that offers lots of interesting features. Because of that, ALPypeRL has certain dependencies to it (e.g. trained policy deployment and evaluation).

.. toctree::
   :maxdepth: 2
   :caption: General:
   
   Home
   AnyLogicConnector
   CartPoleV0
   CartPoleV1
   CartPoleV2
   Evaluation

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
