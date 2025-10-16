"""
Implements the BSW model from Raspopovic et al. (2014) in a growing square domain.

References:
- Raspopovic, Jelena, et al. "Digit patterning is controlled by a 
  Bmp-Sox9-Wnt Turing network modulated by morphogen gradients." 
  Science 345.6196 (2014): 566-570.
"""

# System modules
import sys

# Installed modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

##############
# PARAMETERS #
##############

# Space discretization
nx = 100    # Grid size in x-direction
ny = 100    # Grid size in y-direction
dx = 0.05   # Distance between grid points

# Time discretization
dt = 0.1
totaltime = 1000
time_steps = int(totaltime / dt + 1)

# Model parameters
# See table ST4 in supplementary information of Raspopovic et al. (2014)
# Reaction rate parameters
k2 = 1
k3 = 1
k4 = 1.59
k5 = 0.1
k7 = 1.27
k9 = 0.1
delta = 0

# "BSW model can form a relatively robust Turing pattern with a difference in
# diffusivity of two and a half (d = 2.5) between Bmp and Wnt." - Raspopovic et al. (2014)
Dwnt = 0.001
Dbmp = 2.5*Dwnt 

# Horizontal growth rate
vi = 0.0

# Initial condition parameters for a square domain
# Horizontal: Start on left side of the nx*ny box
Lx0 = 30

max_domain_horizontal = nx # Stop growing once this reached

# Vertical: Start in middle of the nx*ny box
Ly0 = 30


#############
# FUNCTIONS #
#############

def get_domain_size_horizontal(n):
    """
    Calculate the new horizontal domain size.
    Parameters:
        n (int) - timestep
    
    Global variables:
        nx (int)   - horizontal grid size
        Lx0 (int)  - initial horizontal domain size
        dt (float) - time step
        vi (float) - horizontal growth velocity
    """
    return min( round(Lx0 + dt * vi * n), nx)


def single_step_growth(input_matrix, old_bounds, new_bounds):
    """
    Wrapper for growth function.
    """

    if (( old_bounds[1] - old_bounds[0] ) > ( new_bounds[1] - new_bounds[0] )):
        print("Warning: Domain is shrinking. Growth functions do not support shrinking domain.")
        sys.exit()

    # Nothing is changed, but different choices can be made, see later exercises
    output_matrix = input_matrix

    return(output_matrix)


def single_step_RD(sox9, bmp, wnt,                # matrices with variables
                   Dbmp, Dwnt,                    # diffusion coefficients
                   k2, k3, k4, k5, k7, k9, delta, # reaction parameters
                   bounds                         # tissue bounds (minx, maxx, miny, maxy)
                   ):
    """
    Solves a single time step of the reaction-diffusion simulation 
    using explicit forward Euler.
    The diffusion operator (Laplacian) is solved with a 5-point stencil.
    See also: 
    https://en.wikipedia.org/wiki/FTCS_scheme
    https://en.wikipedia.org/wiki/Five-point_stencil
    """
    
     # Get the bounds of the tissue to reduce the area of calculations
    minx, maxx, miny, maxy = bounds #(0, Lx, Lymin, Lymax)

    # DIFFUSION OPERATOR

    # Create a bigger matrix with one extra row and column on each side
    bigL_bmp = np.zeros((bmp.shape[0] + 2, bmp.shape[1] + 2))
    bigL_wnt = np.zeros((wnt.shape[0] + 2, wnt.shape[1] + 2))

    # Implement no-flux boundary condition
    # Fill the domain boundaries with the closest inner value using np.pad
    # -> the gradient is zero -> hence no-flux boundary
    bigL_bmp[minx:maxx+2, miny:maxy+2] = np.pad(bmp[minx:maxx, miny:maxy], 1, mode='edge')
    bigL_wnt[minx:maxx+2, miny:maxy+2] = np.pad(wnt[minx:maxx, miny:maxy], 1, mode='edge')


    # Calculate the Laplacian
    # Create empty matrices for laplacians
    laplacian_bmp = np.zeros_like(bmp)
    laplacian_wnt = np.zeros_like(wnt)  
    # all indices shifted by one to account for extra row and column 
    # so computations are all done on the inner (original) points
    # the extra rows and columns are only used to implement the no-flux boundary condition
    # and saves us from having to explicitly handling these boundary points in the calculations 
    minX = minx + 1
    maxX = maxx + 1     
    minY = miny + 1
    maxY = maxy + 1

    laplacian_bmp[minx:maxx, miny:maxy] = (bigL_bmp[minX+1: maxX+1, minY:maxY] + 
                                           bigL_bmp[minX-1: maxX-1, minY:maxY] + 
                                           bigL_bmp[minX  : maxX  , minY+1:maxY+1] + 
                                           bigL_bmp[minX  : maxX  , minY-1:maxY-1] 
                                           - 4 * bigL_bmp[minX:maxX, minY:maxY])
    
    laplacian_wnt[minx:maxx, miny:maxy] = (bigL_wnt[minX+1: maxX+1, minY:maxY] + 
                                           bigL_wnt[minX-1: maxX-1, minY:maxY] + 
                                           bigL_wnt[minX  : maxX  , minY+1:maxY+1] + 
                                           bigL_wnt[minX  : maxX  , minY-1:maxY-1] 
                                           - 4 * bigL_wnt[minX:maxX, minY:maxY])
    
    # REACTIONS + DIFFUSION
    # Calculate the change in variables in step dt
    dsox9 = np.zeros_like(sox9)
    dbmp  = np.zeros_like(bmp)
    dwnt  = np.zeros_like(wnt)
    
    # Break equations up into multiple lines for readability
    
    # Equation for change in sox9
    dsox9[minx:maxx, miny:maxy] = dt * (k2 * bmp[minx:maxx, miny:maxy] 
                                        - k3 * wnt[minx:maxx, miny:maxy] 
                                        - sox9[minx:maxx, miny:maxy]**3 
                                        + delta * sox9[minx:maxx, miny:maxy]**2)
    
    # Equation for change in bmp
    dbmp[minx:maxx, miny:maxy]  = dt * ((Dbmp / (dx * dx)) * laplacian_bmp[minx:maxx, miny:maxy] 
                                        - k4 * sox9[minx:maxx, miny:maxy] 
                                        - k5 * bmp[minx:maxx, miny:maxy])

    # Equation for change in wnt
    dwnt[minx:maxx, miny:maxy]  = dt * ((Dwnt / (dx * dx)) * laplacian_wnt[minx:maxx, miny:maxy] 
                                        - k7 * sox9[minx:maxx, miny:maxy] 
                                        - k9 * wnt[minx:maxx, miny:maxy])
    
    # Update the variables
    sox9[minx:maxx, miny:maxy] += dsox9[minx:maxx, miny:maxy]
    bmp[minx:maxx, miny:maxy]  += dbmp[minx:maxx, miny:maxy]
    wnt[minx:maxx, miny:maxy]  += dwnt[minx:maxx, miny:maxy]

    return sox9, bmp, wnt


##################
# EXECUTION CODE #
##################

# Check if the simulation parameters respect the FTCS stability criterion.
# See: https://en.wikipedia.org/wiki/FTCS_scheme
if (Dbmp * dt / dx**2) >= 0.25:
    raise ValueError("Stability criterion not met, current value is {}".format(Dbmp * dt / dx**2)) 
else:
    print("Stability criterion met, current value is {}".format(Dbmp * dt / dx**2))


### Define initial condition ###

# Define the initial tissue domain with non-zero values
Lx = get_domain_size_horizontal(0)
Ly = Ly0
Lymin = int(ny/2 - Ly/2)
Lymax = int(ny/2 + Ly/2)

# Allocate matrices for simulation
sox9 = np.zeros((nx, ny))
bmp  = np.zeros((nx, ny))
wnt  = np.zeros((nx, ny))

# Initialize a random value to the tissue domain
sox9[:Lx, Lymin:Lymax] = 0.01 * np.random.randn(Lx, Ly)
bmp[:Lx , Lymin:Lymax] = 0.01 * np.random.randn(Lx, Ly)
wnt[:Lx , Lymin:Lymax] = 0.01 * np.random.randn(Lx, Ly)

# Create a mask for plotting tissue domain (so outside tissue all gets white color)
plot_mask = np.empty((nx, ny))
plot_mask.fill(np.nan)  
plot_mask[:Lx, Lymin:Lymax] = 1

# Set up plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Define the min and max values for each colormap
sox9_vmin, sox9_vmax = -1.5, 1.5  
bmp_vmin, bmp_vmax = -7.5, 7.5
wnt_vmin, wnt_vmax = -10, 10 

# Define custom colors for plots
colors = [(0, 0, 0), (1, 0.2, 0.2)] 
red_dark_cmap = LinearSegmentedColormap.from_list('black_to_red', colors, N=256)
colors = [(0, 0, 0), (0, 1, 0)] 
green_dark_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)
colors = [(0, 0, 0), (0, 1, 1)] 
blue_dark_cmap = LinearSegmentedColormap.from_list('black_to_blue', colors, N=256)


### Simulation loop ###

# A simulation step consists of three events:
#   1. A bigger domain is defined
#   2. A growth function is called that implements growth from the old to the new domain size
#   3. Reaction and diffusion is simulated on the new domain
# The output is plotted every 100 simulation steps.


for timestep in range(time_steps):
    
     # Stop growing if reached maximum size
    if (Lx >= max_domain_horizontal):
        new_bounds = (0, Lx, Lymin, Lymax)

    else:
        # Step 1 -- define the new domain size
        old_Lx = Lx
        Lx = get_domain_size_horizontal(timestep)

        old_bounds = (0, old_Lx, Lymin, Lymax)
        new_bounds = (0, Lx, Lymin, Lymax)

        # Step 2 -- grow the tissue
        bmp  = single_step_growth(bmp,  old_bounds, new_bounds)
        sox9 = single_step_growth(sox9, old_bounds, new_bounds)
        wnt  = single_step_growth(wnt,  old_bounds, new_bounds)

    # Step 3 -- Simulate reaction and diffusion on the domain
    sox9, bmp, wnt = single_step_RD(sox9, bmp, wnt, 
                                    Dbmp, Dwnt, 
                                    k2, k3, k4, k5, k7, k9, delta,
                                    new_bounds)

    ### Plotting ###
    if timestep == 0:  # Initial plot

        current_time = timestep*dt
        
        sox9_plot = ax1.imshow(np.rot90(sox9*plot_mask), cmap=red_dark_cmap, vmin=sox9_vmin, vmax=sox9_vmax)
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot = ax2.imshow(np.rot90(bmp*plot_mask), cmap=green_dark_cmap, vmin=bmp_vmin, vmax=bmp_vmax)
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot = ax3.imshow(np.rot90(wnt*plot_mask), cmap=blue_dark_cmap, vmin=wnt_vmin, vmax=wnt_vmax)
        ax3.set_title(f'wnt at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot

    # Update plot every 100 steps    
    if timestep % 100 == 0:

        # Update the plot mask to the new tissue domain
        plot_mask.fill(np.nan)           # Reset the mask
        plot_mask[:Lx, Lymin:Lymax] = 1  # Label tissue domain only
        
        current_time = timestep*dt

        sox9_plot.set_data(np.rot90(sox9*plot_mask))
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot.set_data(np.rot90(bmp*plot_mask))
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot.set_data(np.rot90(wnt*plot_mask))
        ax3.set_title(f'wnt at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot

plt.show()
