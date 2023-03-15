######################
The AnyLogic Connector
######################

In this page you will learn how to setup your connection from the **AnyLogic side**. To make the explanation more straightforward, we will be using the :ref:`CartPole-v0 example<How to train your first policy. The CartPole-v0 example.>`.

.. note::
    You may find the source code of the *CartPole-v0* `here <https://github.com/MarcEscandell/ALPypeRL/tree/main/alpyperl/examples/cartpole_v0/CartPole_v0>`_.

The summary of steps required to follow are:

1. Add **ALPypeRL Connector Library** to your *AnyLogic Palette*.
2. Drag and drop an instance of **ALPypeConnector** to your **root** (also defaulted as ``Main`` the first time you create a model).
3. Add **ALPypeRLClientController** interface to the list of interfaces in your **root** agent and **implement** the required functions.
4. Call `requestAction` when the RL agent in the simulation requires a new action (or a new instruction) to continue with the run.

**************************************************************
Add **ALPypeRLConnector** library to your **AnyLogic Palette**
**************************************************************

The first thing that you must do is to add the **ALPypeRL Connector Library** to your AnyLogic *Palette*.

If you are not familiar with AnyLogic, this is a pretty straight forward step. You just need to look for the green cross at the end of the *Palette* screen (the location of the *Palette* might vary based on your AnyLogic view arrangement. In general, the *Palette* is presented together with the *Projects* structure and on the left hand side. So you just need to go to the left down corner).

.. image:: images/add_new_library_anylogic_palette.png
    :alt: Add new library to AnyLogic Palette

Once you find it, just click and select `Manage Libraries...`. Then click the `Add` button and select the :download:`ALPypeRLLibrary.jar <../bin/ALPypeRLLibrary.jar>` file.

.. image:: images/add_new_library_anylogic_window.png
    :alt: Add new library to AnyLogic Window

You should now be able to see the newely added library in your list of available libraries at the *Palette*.

.. image:: images/alpyperlconnector_library.png
    :alt: ALPypeRL library in Palette

**************************************************
Drag and drop an instance of **ALPypeRLConnector**
**************************************************

Now that you have access to the **ALPypeRL Connector** from AnyLogic, you can proceed to drag and drop and instance of it into your model.

Here it is very important that you place the connector in your **root** agent. If you are not familiar with AnyLogic, the root agent is the one that normally holds and compiles the rest of the objects in your simulation (it's like the *home* for everything else).

.. tip:: 
    Another reference that you can take, it's the agent that you select when you set up your ``Simulation`` experiment as the *Top-level agent*. See the image below:
    
    .. image:: images/root_agent.png
        :alt: AnyLogic root agent

************************************
Implement `ALPypeRLClientController`
************************************

This is a very important step in order for the *ALPypeRL Connector* to understand what it needs to do when the training starts or when you are evaluating your policy.

First, you must add ``ALPypeRLClientConnector`` to the list of interfaces of your **root** agent. If you are not familiar with AnyLogic, you can find it by: first click on a random point in the canvas of your root agent (also known as ``Main``) and then navigate to the *Properties* page. Once you are there, you must scroll down and find the section *Advanced Java*. In there, you should be able to see *Implements (comma-separated list of interfaces)*. Then you can add ``ALPypeRLClientController``.

.. image:: images/root_interface.png
    :alt: Root interface

Next, if you try to compile your model, you will be getting at least 4 new errors as shown in the image:

.. image:: images/interface_errors.png
    :alt: Interface error

This is basically telling you that, if you plan to implement that class, you must implement those functions (it's kind of a contract that you have decided to sign).

You can now drag and drop 4 new functions. Their arguments and return types must be as follows (otherwise the compilation error won't go away):

* ``void takeAction(ActionSpace action)``: Here you must tell the model what to do or how to apply the action that is comming as an argument. 

    .. note::
        The **action type must match** what you have (or will define) in your **python script**. Refer to :ref:`Gym Action and Observation spaces <Create an *Action* and *Observation* spaces>`.

* ``double[] getObservation()``: Return the observation seen at that moment in time in the form of a ``double[]`` array.

* ``double getReward()``: Return the reward observed at that moment in time.

    .. warning:: Note that this should not be a cumulated value (e.g. in the *CartPole-v0* example, the cart gets a reward of 1 for every step that manages to keep the pole straight and within boundaries).

* ``boolean hasFinised()``: Return ``true`` if any custom rule that required the simulation to stop has been met (e.g. the pole attached to the cart has exceeded a certain non-recoverable angle or the simulation has reached the end).

.. image:: images/interface_impl.png
    :alt: Interface implementation

******************************************************************************
Call `requestAction` when the RL agent in the simulation requires a new action
******************************************************************************

.. important::
    In this last step, you simply must **call** ``requestAction()`` **at the location where your agent will need to receive an action** so it can proceed.

The function is accessible from the ``alPypeRLConnector`` instance (e.g. ``alPypeRLConnector.requestAction()``).

In the *CartPole-v0* example, there is a cyclic event that updates the status of the system (*horizontal positon*, *cart speed*, *pole angle* and *pole angular velocity*). At that moment in the simulation, the cartpole is requesting the next action: whether to apply a force on the right or the left.

.. image:: images/event_request_action.png
    :alt: requestAction() function