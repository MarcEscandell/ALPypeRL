from setuptools import setup, find_packages

# Read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='alpyperl',
      version='1.1.2',
      license='Apache License, Version 2.0',
      author='Marc Escandell Mari',
      author_email='marcescandellmari@gmail.com',
      description='An open source library for connecting AnyLogic models with Reinforcement Learning frameworks through OpenAI Gymnasium ',
      long_description=long_description,
      long_description_content_type="text/markdown",
      python_requires='>=3.9, <3.11',
      packages=find_packages(exclude=['alpyperl.examples', 'alpyperl.examples.*', 'tests', 'tests.*']),
      url='https://github.com/MarcEscandell/ALPypeRL',
      keywords='alpyperl',
      install_requires=[
            'gymnasium==0.28.1',
            'py4j==0.10.9.7',
            'ray[rllib]==2.31.0',
            'tensorflow==2.16.2',
            'torch==2.3.1',
            'uvicorn==0.30.1',
            'fastapi==0.111.0'
      ],
      extras_require={
            'docs': [
                'sphinx_rtd_theme'
            ]
      }
)