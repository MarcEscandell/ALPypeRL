import logging
from py4j.clientserver import ClientServer, JavaParameters, PythonParameters
from threading import Event, Thread
import socket
from alpyperl.anylogic.model.launcher import ALModelLauncher


def get_open_port():
    """This method allows finding free ports in host machine"""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("",0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


class AnyLogicModelCallback:
    """This is the implementation of the interface that has been defined in the
    Java side and allows AnyLogic to know what functions is allowed to access
    """
    # -----------------------------------------------------------------
    # Code here (any method created here needs to be added to the AL
    # interface) and must be named using the same structure
    # -----------------------------------------------------------------
    def __init__(self):
        self.thread_handler = None

    def finishedModelSetup(self):
        """This function is called from the Java side to unblock python script
        which is waiting for the simulation to finish loading and setting up
        """
        self.thread_handler.set()
        self.thread_handler.clear()
        return True

    def toString(self):
        """Need 'toString()' implementation because AnyLogic calls this
        to visualise the value of variables
        """
        return "Python interface controller for AnyLogic model in java"

    # It is required to define the Java interface that is implemented. Note that is
    # important to properly define the package name as in AnyLogic
    class Java:
        implements = ["com.alpyperl.RLPythonController"]


class AnyLogicModelConnector:
    """This is the main class that creates the gateway between the python script
    and the AnyLogic model. As well as in charge of executing and setting the
    simulation
    """

    def __init__(
        self,
        run_exported_model,
        exported_model_loc,
        show_terminals
    ):
        self.logger = logging.getLogger(__name__)
        # Initialise model launcher
        self.al_model_launcher = None
        # Create an instance of the python implementation to be accessed by the
        # AnyLogic model
        self.anylogic_model_callback = AnyLogicModelCallback()

        # Create single thread gateway between python and AnyLogic (only possible with
        # ClientServer)
        # We want it to be single-thread to force things to happen in a certain
        # order as defined by the python script and not in parallel (asyncronously)
        #
        # E.g. if python script takes some seconds to calculate a decision, the simulation
        # should be waiting for it to finish because it requires the result of the
        # operation to continue (e.g. next action)

        # Dynamic ports is necessary to allow multiple runs to be executed in
        # parallel.
        # This wouldn't be possible using the default configuration because it always
        # uses the same default ports (Java port: 25333, Python port: 25334)

        # This is the port that will be used from the Java side
        java_port = get_open_port() if run_exported_model else 25333

        # Connect python side to Java side with Java dynamic port and start python
        # callback server with a dynamic port (port=0)
        self.gateway = ClientServer(
            java_parameters=JavaParameters(auto_field=True, port=java_port),
            python_parameters=PythonParameters(port=0 if run_exported_model else 25334),
            python_server_entry_point=self.anylogic_model_callback
        )
        # Retrieve the port on which the python callback server was bound to.
        python_port = (
            self.gateway.get_callback_server().get_listening_port()
        )
        self.logger.debug(
            "ALPypeRL connection will take place on ports "
            f"(Java: {java_port} / Python: {python_port})"
        )
        # Execute AnyLogic anylogic_model and tell to which ports it needs to
        # connect. All of this will be handled by the 'ALModelLauncher'
        def execute_model():
            self.al_model_launcher = ALModelLauncher(
                java_port=java_port,
                python_port=python_port,
                folder_location=exported_model_loc,
                show_terminals=show_terminals
            )
            self.al_model_launcher.compile_and_run()
        # Run the model in another thread to avoid this being called from the same thread
        # as the python script, which will cause the execution to be on hold (due to the
        # AL model waiting for instructions - listening)
        if run_exported_model:
            thread = Thread(target=execute_model, args=[])
            thread.start()
        else:
        # In case the user wants to run the model from AnyLogic directly (not 
        # using the exported version) the script will notify him/her that the
        # python script is waiting and ready for the AnyLogic model to be launched
            self.logger.info(
                "You can now launch your AnyLogic model! "
                "'ALPypeRLConnector' will handle the connection for you."
            )
        # The following code will block any further execution of the python code
        # until the AnyLogic model calls back and unblocks the event via the
        # python interface
        thread_handler = Event()
        self.anylogic_model_callback.thread_handler = thread_handler
        self.logger.debug(
            "Python AnyLogic connection handler is waiting AnyLogic model side "
            "to launch and connect"
        )
        thread_handler.wait()

    def close_connection(self):
        """Close model and connection"""
        if self.al_model_launcher is not None:
            # First, close gateway
            self.gateway.shutdown()
            # Then, close model
            self.al_model_launcher.close_model()

    def __del__(self):
        """Destructor"""
        self.close_connection()
