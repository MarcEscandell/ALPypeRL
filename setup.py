from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='alpyperl',
      version='0.0.13',
      license='Apache License, Version 2.0',
      author='Marc Escandell Mari',
      author_email='marcescandellmari@gmail.com',
      description='An open source library for connecting AnyLogic models with Reinforcement Learning frameworks through OpenAI Gymnasium ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=find_packages(),
      url='https://github.com/MarcEscandell/ALPypeRL',
      keywords='alpyperl',
      install_requires=[
            'gymnasium',
            'py4j',
            'ray',
            'ray[rllib]',
            'tensorflow',
            'torch',
            'uvicorn',
            'fastapi'
      ]
)