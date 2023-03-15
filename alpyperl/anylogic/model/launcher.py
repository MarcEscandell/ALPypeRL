import os
import platform
import re
import string
import random
import subprocess
from subprocess import Popen
from pathlib import Path
import logging


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
        self.exec_loc = None
        self.al_process = None
    
        # Create command-line arguments that refer to the Java and Python ports
        # java model needs to connect to.
        self.port_arg_str = '-jp ' + str(java_port) + ' -pp ' + str(python_port)
        # Get project name.
        self.project_name = self.__get_project_name(folder_location=folder_location)

    def compile_and_run(self):
        """Compile the script and execute it considering the operating system
        it is running on"""

        # Execute exported version of the model depending on OS
        if self.os_name == 'Linux':
            # Generate executable file
            self.exec_loc = f"./{self.__create_exec_file('linux.sh')}"
            # Execute model
            self.al_process = (
                Popen(['gnome-terminal', '--', self.exec_loc]) 
                if self.show_terminals 
                else Popen(['bash', '-c', self.exec_loc])
            )

        elif self.os_name == 'Windows':
            # Generate executable file
            self.exec_loc = self.__create_exec_file('windows.bat')
            # Execute model
            self.al_process = (
                Popen(['start', 'cmd.exe', '/c', self.exec_loc], shell=True)
                if self.show_terminals 
                else Popen([self.exec_loc])
            )

        elif self.os_name == 'Darwin':
            # Generate executable file
            self.exec_loc = self.__create_exec_file('mac')
            # Execute model
            self.al_process = (
                Popen(['open', self.exec_loc])
                if self.show_terminals 
                else Popen([f'./{self.exec_loc}'])
            )
        self.logger.debug(
            f"AnyLogic model '{self.project_name}' has been succesfully "
            "compiled and launched"
        )

    def close_model(self):
        """ Delete model executable file and close process"""
        os.remove(self.exec_loc)
        self.al_process.terminate()
        self.logger.debug("AnyLogic models have been terminated")

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
            os.system(f'chmod +x {exec_loc}')

        return exec_loc
