# Reinforcement Learning basics

If you are new to Reinforcement Learning, in this page you'll learn some basics. Although, the best recommendation is to visit the [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/). It's probably the best free **educational** resource at the moment if you want to learn in deep detail how RL works.

As explained in [wikipedia](https://en.wikipedia.org/wiki/Reinforcement_learning):

> **Reinforcement learning (RL)** is an area of **machine learning** concerned with how **intelligent agents** ought to **take actions** in an **environment** in order to **maximize** the notion of cumulative **reward**. Reinforcement learning is one of three basic machine learning paradigms, alongside supervised learning and unsupervised learning.

> Reinforcement learning differs from supervised learning in not needing labelled input/output pairs to be presented, and in not needing sub-optimal actions to be explicitly corrected. Instead the focus is on finding a balance between exploration (of uncharted territory) and exploitation (of current knowledge).


![RL diagram from wikipedia](images/rl_diagram.svg)

If we want to relate what has been explained to ALPypeRL we can say that:

* The **intelligent agents that take decisions** are the _policies_ trained (e.g. using `rllib`). You will need the python package **alpyperl** for dealing with agents. Agent/Policy learning happens in ALPypeRL on the _python_ side.
* The **environment** that is used as the _playground_ for the policy to learn from via **observation** collection happens on the AnyLogic side. This is all connected thanks to the **ALPypeRLConnector** and the implementation of the required **ALPypeRLClientController** functions. A **reward** will be generated after taking an action. Then, the agent will try to maximize its cumulative value.

Other references:

* [RLlib](https://docs.ray.io/en/master/rllib/core-concepts.html)
* [AnyLogic](https://www.anylogic.com/features/artificial-intelligence/)

Next, [how to setup the AnyLogic Connector](https://github.com/MarcEscandell/alpyperl/wiki/AnyLogicConnector).