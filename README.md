# CLS-thesis
Constructing digital twin of Amsterdam social network

This project shows the progress of the construction of the digital twin of Amsterdam.


## Folder description

`Data` the Data folder is a folder which consists of the CBS Data and several folders.

- The CBS data comprises 5 main files:
    - tab_buren: which is the neighbours data.
    - tab_familie: which is the family data.
    - tab_huishouden: which is the household data.
    - tab_werkschool: which is the work/school data.
    - tab_n: Which is the data of the population of Amsterdam.
    - tab_n_(with oplniv): Which is the data of the population of Amsterdam with education level.

`Figures` The Figures folder include all the figures per dataset, whcih shows the heatmaps and Distribution of the chosen dataset.

`Notebooks` The Notebook Folder include some experimental notebooks.

`Papers` The Paper folder includes relevant books and papers for particular subjects.

`Presentations` Are the weekly presentations.


## Script descriptions

`descriptive.py` is a python script which generates all the figures and distributions found in the Figures folder, it also has some other descriptive functions.

`education.py` is a script which makes the converts the tab_n data to tab_n_(with oplniv) based on the connections each group has in each layer of the network.

`main.py` is the file where the static network is created.

`segregation.py` is a file where the isolation and seregation index are calculated from the data

`Network_analysis.ipynb` is a notebook where the network analysis is conducted