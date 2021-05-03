# Travelling Salesman Problem
Repository containing work completed as part of final year module; Theoretical Aspects of Computer Science


### Members
| Name          								|
| -------------									|
| Jack Allen    								|
| Antonio Cardoso Queiroz da Silva Roque      	|
| Gheorghe Craciun								|
| Francisco Gaspar Ramos						|
| Goncalo Lima Carvalheda						|


## Dependencies
* [Anaconda](https://docs.anaconda.com/anaconda/install/)
* [Python 3.9.1](https://www.python.org/downloads/)
* [Jupyter](https://jupyter.org/install)
* [Numpy](https://numpy.org/install/)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
* [Plotly](https://plotly.com/python/getting-started/#installation)


### SETUP
Please follow the below instructions to set up an Anaconda environment which contains project requirements for running jupyter notebook.

**Note that instructions assume you have already cloned the repo to your local machine.**
 #### Create anaconda environment

1) Install [Anaconda](https://docs.anaconda.com/anaconda/install/)
2) In a terminal located at the base of the repository directory (`380CT_Assignment/`), once Anaconda is installed, create a python virtual environment using the following command:
	* `conda create --name TAoCS_380CT python=3.9 --file requirements.txt`
3) You can activate or deactivate the environment using the following commands:
	* Activate: `conda activate TAoCS_380CT`
	* Deactivate:  `conda deactivate`
4) If you haven't already, activate the conda environment: `conda activate TAoCS_380CT`
5) Your all set to run the Jupyter notebook! Enter the following command within your virtual environment from `380CT_Assignment/Jupyter_Notebook/`:   
	* `jupyter notebook 380CT_Notebook.ipynb`



## Workflow

### Updating Conda Environment

####  Add Dependencies to Project
If you install a new dependency into conda environment that should be accessible for development, please make sure you add it to the requirements.txt file using the following command:

* `conda list -e > requirements.txt`

**PLEASE NOTE* that when installing any dependencies to use `conda install` and NOT `pip`. A quick google search will help you find the conda equivalent. 

This is important because for below command updating the environment using requirements.txt will fail to read pip (not impossible, but altering the workflow to cater for this is outside the scope of this project).


#### Update Environment from requirements.txt
If you need to update your anaconda environment to reflect new dependencies added to requirements.txt, use the following command:

* `conda install --file requirements.txt`
