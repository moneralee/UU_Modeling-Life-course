"""Parameter file"""

###############################

# installed modules
import numpy as np

###############################
###############################
# Simulation parameters

init_MCS  = 0     # Monte Carlo Steps of initialization routine
total_MCS = 3000  # total number of Monte Carlo Steps

# Grid size
height = 70 # number of rows
width  = 70 # number of cols

initial_cell_number = [25, 25]

initial_cell_config = "hex_grid" # "point" or "hex_grid"

# kT
"""
The cellular Potts model was originally inspired by the
Potts model of statistical mechanics, where the
Boltzmann constant times temperature (kT) is a
parameter that determines the energy scale of the
system. In the context of the cellular Potts model,
kT is used to control the motility or activity of the cells
and hence loses its physical meaning.
"""
kT = 10

# ADHESION
"""
The adhesion table stores pairwise adhesion values
between different cell types (tau).

Interaction mapping:
tau1:  0   1   2  
tau2:
   0  0-0 0-1 0-2 
   1  1-0 1-1 1-2
   2  2-0 2-1 2-2

It is a diagonally symmetric matrix.
Only the lower diagonal needs to be defined, 
the rest is auto-filled.

The adhesions are expressed as affinities, and in
the code their sign is flipped. This means that a
more positive value indicates stronger adhesion.
Stronger means more energetically favorable.
A negative value indicates repulsion, or energetically
not favorable.
"""

adhesion_table = np.array([
    [     0,   0,    0], # adhesion medium - partner
    [   -16,  -2,    0], # adhesion cell type 1 - partner
    [   -16, -11,  -14], # adhesion cell type 2 - partner
])

# Make the matrix diagonally symmetric (copy lower triangle to upper triangle)
i_lower = np.tril_indices_from(adhesion_table, -1)
adhesion_table[i_lower[::-1]] = adhesion_table[i_lower]


# Target volume (or area in 2D)
"""Target volume / area"""
target_volume = np.array([
    40, # target volume of cell type 1
    40, # target volume of cell type 2
])

# lambda is a Lagrange multiplier or penalty factor
lambda_volume = np.array([
    1, # cell type 1
    1, # cell type 2
])


# Surface area (or perimeter in 2D)
"""Target surface area / perimeter"""
target_surface = np.array([
    55, # target surface of cell type 1
    55, # target surface of cell type 2
])

# lambda is a Lagrange multiplier or penalty factor
lambda_surface = np.array([
    1,  # cell type 1
    1,  # cell type 2
])

###############################
###############################
# Graphics parameters

gui_scale = 4 # scale resolution

gui_update_frequency = 10 # how often the GUI is updated in time steps

gui_cell_colors = {0: (250, 250, 250), # medium
                   1: (24, 111, 194),   # cell type 1
                   2: (182, 77, 184),   # cell type 2
                   9999999: (180, 180, 180) # last index reserved for colour of cell boundary
}