## A Python implementation of SAFE toolbox for sensitivity analysis

Pianosi, F., Sarrazin, F., Wagener, T. (2015), A Matlab toolbox for Global Sensitivity Analysis, Environmental Modelling & Software, 70, 80-85.
 
https://www.safetoolbox.info


## Installating the package

# Option 1: from python command line (e.g. Anaconda prompt), go into the SAFEpython_v0.0.0 folder and execute:

    pip install .

# Option2: otherwise from command line go into the SAFEpython_v0.0.0 folder and execute:

	python -m pip install .


Notes for Windows users: python cannot be called directly from Windows command line. You have to go into the folder in wich python is installed and then execute:

	python -m pip install mydir\SAFEpython_v0.0.0

(mydir is the directory in which the SAFEpython_v0.0.0 folder is saved, and it shoud not contain which spaces)


# NOTE: to install the package without administrator rights, you may have to use:
    pip install --user mydir\SAFEpython_v0.0.0


## Getting started with the package

We recommend to start by executing the workflow scripts that apply the different global sensitivity analysis methods to test cases. 
