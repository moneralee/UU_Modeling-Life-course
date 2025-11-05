# Cellular Potts Model

This folder contains an implementation of the CPM that was forked from source code provided with the publication of [Bao et al., Nature Cell Biology 2022](https://doi.org/10.1038/s41556-022-00984-y). 
The code was refactored to be more modular and was extended with a basic graphical user interface for real-time visualization using PyQt5.
The GitHub repository with the original code can be found at: https://github.com/jakesorel/CPM_ETX_2022

## Code Structure

The code is organized as follows:

- The file "main.py" is the entry-point script that should be executed to run the simulation.
- The file "parameters.py" is used to define parameters used for the CPM simulation and for the visualization.
- The subfolder "CPM" contains the core simulation code forked from [jakesorel/CPM_ETX_2022](https://github.com/jakesorel/CPM_ETX_2022).
- The subfolder "GUI" contains a simple PyQt5-based visualization.