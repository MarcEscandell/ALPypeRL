import os
import platform
import re
import string
import random
import subprocess
from subprocess import Popen
from pathlib import Path
import logging
import signal
import shlex


class ALModelLauncher():
    """This class in in charge of launching the AnyLogic model. The main
    functionality comes by notifying the model what ports need to be connecting
    to so communication with python script is possible.
    """

    def __init__(
        self,
        java_port=25333,
        python_port=25334,
        folder_location='./exported_model',
        show_terminals=False
    ):
        self.logger = logging.getLogger(__name__)
        # Check if exported model exists.
        if not os.path.exists(folder_location):
            raise Exception(
                f"Could not find exported model folder '{folder_location}'. "
                "Model execution could not proceed. Please export your model at "
                f"'{folder_location}'"
            )
        # Get operating system compiling this code
        self.os_name = platform.system()
        self.folder_location = Path(folder_location)
        self.show_terminals = show_terminals
        self.java_exec_end_pattern = '%' if self.os_name == 'Windows' else '$'
        # Initialise executable file location and process
        self.executable_location = None
        self.al_process = None
    
        # Create command-line arguments that refer to the Java and Python ports
        # java model needs to connect to.
        self.port_arg_str = '-jp ' + str(java_port) + ' -pp ' + str(python_port)
        # Get project name.
        self.project_name = self.__get_project_name(folder_location=folder_location)

    def compile_and_run(self):
        """Compile the script and execute it considering the operating system."""
        # Determine file extension based on OS
        file_extension = {
            'Linux': '.sh',
            'Windows': '.bat',
            'Darwin': ''  # No extension for Mac
        }.get(self.os_name, '')

        # Adjust executable name based on OS
        os_name_for_exec = 'mac' if self.os_name == 'Darwin' else self.os_name.lower()
        self.executable_location = str(self.__create_exec_file(f'{os_name_for_exec}{file_extension}'))

        # Enclose in quotes to handle spaces
        quoted_executable = (
            f'"{self.executable_location}"' if self.os_name == 'Windows' 
            else shlex.quote(self.executable_location)
        )

        # Define command based on OS
        command = {
            'Linux': (
                ['bash', '-c', quoted_executable] 
                if not self.show_terminals 
                else ['gnome-terminal', '--', self.executable_location]
            ),
            'Windows': quoted_executable,
            'Darwin': (
                ['open', quoted_executable] 
                if self.show_terminals 
                else [quoted_executable, 'preexec_fn=os.setsid']
            )
        }.get(self.os_name)

        # Execute model
        self.al_process = Popen(
            command,
            shell=(self.os_name == 'Windows'),
            start_new_session=True
        )
        self.logger.debug(f"AnyLogic model '{self.project_name}' has been successfully compiled and launched.")



    def close_model(self):
        """ Delete model executable file and close process"""
        try:
            # Delete executable file traces
            os.remove(self.executable_location)
            # TODO: For now terminals must be closed manually since PID process is
            # unkown. This is because the process is launched in a new terminal
            # and tracking is lost
            if self.show_terminals:
                self.al_process.terminate()
            else:
                os.killpg(os.getpgid(self.al_process.pid), signal.SIGTERM)
            self.logger.debug("AnyLogic models have been terminated")

        except Exception as e:
            self.logger.error(f"Error during model termination: {e}")

    def __get_project_name(self, folder_location):
        """ A function to extract the AnyLogic project name automatically based on
        the expected files to be found after exporting the model to a standalone
        """
        for f in os.listdir(folder_location):
            matches = re.search('(.*)_linux\.sh', f)
            if matches:
                return matches.group(1)

        raise Exception(
            f"Could not find any executable file in '{folder_location}'. "
            "Have you exported your model correctly?"
        )

    def __id_generator(self, size=6, chars=string.ascii_uppercase + string.digits):
        """This is to avoid re-writting the an active script. It can happen when
        asyncronous execution"""
        return ''.join(random.choice(chars) for _ in range(size))

    def __create_exec_file(self, file_type):
        """Insert ports information and create executable file"""
        # Get file extension
        file_extension = Path(file_type).suffix
        # Read file as string
        with open(self.folder_location.joinpath(f'{self.project_name}_{file_type}'), 'r') as file:
            orig_exec_file_str = file.read()
        # Apply regex to insert ports information
        exec_file_str = re.sub(
            f'([\s\S]*)(\{self.java_exec_end_pattern}\*)([\s\S]*)',
            f'\g<1>{self.port_arg_str} \g<2>\g<3>',
            orig_exec_file_str
        )
        # File name and location
        exec_loc = self.folder_location.joinpath(f'{self.project_name}-{self.__id_generator()}{file_extension}')
        # Create and write bash script
        with open(exec_loc, "w+") as file:
            file.write(exec_file_str)
        # Parse text file to unix executable extension only if in Linux system
        if not self.os_name == 'Windows':
            os.system(f"chmod +x '{exec_loc}'")

        return exec_loc
