"""
Implements the BSW model from Raspopovic et al. (2014) in a square domain.

References:
- Raspopovic, Jelena, et al. "Digit patterning is controlled by a 
  Bmp-Sox9-Wnt Turing network modulated by morphogen gradients." 
  Science 345.6196 (2014): 566-570.
"""

# Installed modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

##############
# PARAMETERS #
##############

# Grid and simulation parameters
nx = 100    # Grid size in x-direction
ny = 100    # Grid size in y-direction
dx = 0.05   # Distance between grid points

# Time discretization
dt = 0.1
totaltime = 600
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


#############
# FUNCTIONS #
#############

def single_time_step(sox9, bmp, wnt,               # matrices with variables
                     Dbmp, Dwnt,                   # diffusion coefficients
                     k2, k3, k4, k5, k7, k9, delta # reaction parameters
                     ):
    
    """
    Solves a single time step of the reaction-diffusion simulation 
    using explicit forward Euler.
    The diffusion operator (Laplacian) is solved with a 5-point stencil.
    See also: 
    https://en.wikipedia.org/wiki/FTCS_scheme
    https://en.wikipedia.org/wiki/Five-point_stencil
    """
    
    # DIFFUSION OPERATOR
    laplacian_bmp = np.zeros_like(bmp)
    laplacian_wnt = np.zeros_like(wnt)
    laplacian_bmp[1:-1, 1:-1] = (bmp[2:, 1:-1] + bmp[:-2, 1:-1] + 
                                 bmp[1:-1, 2:] + bmp[1:-1, :-2] - 4 * bmp[1:-1, 1:-1])
    laplacian_wnt[1:-1, 1:-1] = (wnt[2:, 1:-1] + wnt[:-2, 1:-1] + 
                                 wnt[1:-1, 2:] + wnt[1:-1, :-2] - 4 * wnt[1:-1, 1:-1])
    bmp[0, :] = bmp[1, :]
    bmp[nx-1, :] = bmp[nx-2, :]
    bmp[:, 0] = bmp[:, 1]
    bmp[:, ny-1] = bmp[:, ny-2]
    
    wnt[0, :] = wnt[1, :]
    wnt[nx-1, :] = wnt[nx-2, :]
    wnt[:, 0] = wnt[:, 1]
    wnt[:, ny-1] = wnt[:, ny-2]
    
    # REACTIONS + DIFFUSION
    # Calculate the change in variables in step dt
    dsox9 = dt*( k2*bmp - k3*wnt - sox9**3 + delta*sox9**2 )
    dbmp  = dt*( (Dbmp/(dx*dx))*laplacian_bmp - k4*sox9 - k5*bmp )  
    dwnt  = dt*( (Dwnt/(dx*dx))*laplacian_wnt - k7*sox9 - k9*wnt ) 
    
    # Update the variables
    sox9 +=dsox9
    bmp  +=dbmp
    wnt  +=dwnt

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

# Initialize matrices for simulation with random initial condition
sox9 = 0.01 * np.random.randn(nx, ny)
bmp  = 0.01 * np.random.randn(nx, ny)
wnt  = 0.01 * np.random.randn(nx, ny)

# Set up plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Define the min and max values for each colormap
sox9_vmin, sox9_vmax = -2.5, 2.5  
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

for timestep in range(time_steps):

    # Simulate one step of reaction and diffusion
    sox9, bmp, wnt = single_time_step(sox9, bmp, wnt, 
                                      Dbmp, Dwnt, 
                                      k2, k3, k4, k5, k7, k9, delta)
    
    ### Plotting ###
    if timestep == 0:  # Initial plot
        
        current_time = timestep*dt

        sox9_plot = ax1.imshow(sox9, cmap=red_dark_cmap, vmin=sox9_vmin, vmax=sox9_vmax)
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot = ax2.imshow(bmp, cmap=green_dark_cmap, vmin=bmp_vmin, vmax=bmp_vmax)
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot = ax3.imshow(wnt, cmap=blue_dark_cmap, vmin=wnt_vmin, vmax=wnt_vmax)
        ax3.set_title(f'wnt at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot

    # Update plot every 100 steps    
    elif timestep % 100 == 0:
        
        current_time = timestep*dt

        sox9_plot.set_data(sox9)    
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot.set_data(bmp)
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot.set_data(wnt)
        ax3.set_title(f'wnt at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot

        if timestep % totaltime/10 == 0:

            min_sox9, max_sox9 = np.min(sox9), np.max(sox9)
            min_bmp, max_bmp = np.min(bmp), np.max(bmp)
            min_wnt, max_wnt = np.min(wnt), np.max(wnt)

            print()
            print("Time step", current_time)
            print(f"min and max of sox9: {min_sox9}, {max_sox9}")
            print(f"min and max of bmp: {min_bmp}, {max_bmp}")
            print(f"min and max of wnt: {min_wnt}, {max_wnt}")

plt.show()