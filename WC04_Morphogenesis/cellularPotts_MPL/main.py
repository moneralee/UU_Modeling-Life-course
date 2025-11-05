"""
This is the main script to run to execute the program.
"""

#################################

# Local modules
import parameters as PAR
from simulation import cpm as CPM
from graphics import basic_gui as GUI

#################################

def set_simulation_params():
    """
    Convert from the parameter names defined in the input file to a dictionary of parameters.
    """

    params = {
              "nx"         : PAR.width,
              "ny"         : PAR.height,
              "init_cells" : PAR.initial_cell_number,
              "init_conf"  : PAR.initial_cell_config,
              "A0"         : PAR.target_volume,
              "P0"         : PAR.target_surface,
              "lambda_A"   : PAR.lambda_volume,
              "lambda_P"   : PAR.lambda_surface,
              "W"          : PAR.adhesion_table,
              "T"          : PAR.kT
             }

    return(params)


def set_graphics_params():
    """
    Convert from the parameter names defined in the input file to a dictionary of parameters.
    """

    params = {  
              "scale" : PAR.gui_scale,
              "freq"  : PAR.gui_update_frequency,
              
              "colors" : PAR.gui_cell_colors,
              
              "t_tot" : PAR.total_MCS,
              "t_ini" : PAR.init_MCS

             }

    return(params)

#################################

if __name__ == "__main__":

    print("Launching CPM simulation...")

    # Define simulation
    sim_params = set_simulation_params()

    cpm = CPM.CPM(sim_params)
    
    # Start the GUI and run the simulation
    gui_params = set_graphics_params()
    window = GUI.Visualization(gui_params, cpm)
    window.run()