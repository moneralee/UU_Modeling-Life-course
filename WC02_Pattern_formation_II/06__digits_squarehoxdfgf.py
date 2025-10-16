"""
Implements the BSW model from Raspopovic et al. (2014) in a growing square domain.

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
nx = 35    # Grid size in x-direction
ny = 35    # Grid size in y-direction
dx = .05   # Distance between grid points

# Time discretization
dt = 0.1
totaltime = 2000
time_steps = int(totaltime / dt + 1) 

# Model parameters
# See table ST4 in supplementary information of Raspopovic et al. (2014)
# Reaction rate parameters
k2 = 1
k3 = 1
k4 = 1.95  # NOTE new value! old: 1.59
k5 = 0.1
k7 = 0.55  # NOTE new value! old: 1.27
k9 = 0.1
delta = 0 #2
k_HF_bmp = 0.36
k_HF_wnt = 0.72
# Degradation and diffusion of fgf
k_deg_fgf = 0.0003 
Dfgf = 0.001  


# "BSW model can form a relatively robust Turing pattern with a difference in
# diffusivity of two and a half (d = 2.5) between Bmp and Wnt." - Raspopovic et al. (2014)
Dwnt = 0.001
Dbmp = 2.5*Dwnt 
  

p_HF_hox = 0.5
p_HF_fgf = 0.5
k_deg_hox = 0.1
k_deg_fgf = 0.1
Dhox = 0.0  
Dfgf = 0.0  


#############
# FUNCTIONS #
#############

def single_time_step(sox9, bmp, wnt, fgf, hox,     # matrices with variables
                     Dbmp, Dwnt, Dfgf,        # diffusion coefficients
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
    
    # In this case, the domain boundaries are the full size of the matrix
    minx = 0
    maxx = sox9.shape[0]
    miny = 0
    maxy = sox9.shape[1]

    # DIFFUSION OPERATOR

    # Create a bigger matrix with one extra row and column on each side
    bigL_bmp = np.zeros((bmp.shape[0] + 2, bmp.shape[1] + 2))
    bigL_wnt = np.zeros((wnt.shape[0] + 2, wnt.shape[1] + 2))
    

    # Implement no-flux boundary condition
    # Fill the domain boundaries with the closest inner value
    # -> the gradient is zero -> hence no-flux boundary
    bigL_bmp[minx:maxx+2, miny:maxy+2] = np.pad(bmp[minx:maxx, miny:maxy], 1, mode='edge')
    bigL_wnt[minx:maxx+2, miny:maxy+2] = np.pad(wnt[minx:maxx, miny:maxy], 1, mode='edge')
    

    # Calculate the Laplacian

    # Create empty matrices for laplacians
    laplacian_bmp = np.zeros_like(bmp)
    laplacian_wnt = np.zeros_like(wnt)  
    

    # all indices shifted by one to account for extra row and column
    minX = minx + 1
    maxX = maxx + 1     
    minY = miny + 1
    maxY = maxy + 1

    laplacian_bmp[0:nx, 0:ny] = (bigL_bmp[minX+1: maxX+1, minY:maxY] + 
                                 bigL_bmp[minX-1: maxX-1, minY:maxY] + 
                                 bigL_bmp[minX  : maxX  , minY+1:maxY+1] + 
                                 bigL_bmp[minX  : maxX  , minY-1:maxY-1] 
                                 - 4 * bigL_bmp[minX:maxX, minY:maxY])
    
    laplacian_wnt[0:nx, 0:ny] = (bigL_wnt[minX+1: maxX+1, minY:maxY] + 
                                 bigL_wnt[minX-1: maxX-1, minY:maxY] + 
                                 bigL_wnt[minX  : maxX  , minY+1:maxY+1] + 
                                 bigL_wnt[minX  : maxX  , minY-1:maxY-1] 
                                 - 4 * bigL_wnt[minX:maxX, minY:maxY])
    
       
    # REACTIONS + DIFFUSION
    # Calculate the change in variables in step dt
    dsox9 = dt*( k2*bmp - k3*wnt - sox9**3 + delta*sox9**2 )
    dbmp  = dt*( (Dbmp/(dx*dx))*laplacian_bmp - (k4 - k_HF_bmp*fgf*hox)*sox9 - k5*bmp )  
    dwnt  = dt*( (Dwnt/(dx*dx))*laplacian_wnt - (k7 + k_HF_wnt*fgf*hox)*sox9 - k9*wnt )
    

    # Update the variables
    sox9 += dsox9
    bmp  += dbmp
    wnt  += dwnt

    return sox9, bmp, wnt, fgf


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

# Define the initial non-zero values
sox9 = 0.01 * np.random.randn(nx, ny)
bmp  = 0.01 * np.random.randn(nx, ny)
wnt  = 0.01 * np.random.randn(nx, ny)
# Hox and Fgf are initialised at zero
hox  = np.zeros((nx, ny))
fgf  = np.zeros((nx, ny))   

# Set up plots
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
ax1, ax2, ax3, ax4, ax5, ax6 = axs.flatten()

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
for timestep in range(time_steps):

    #Note that this has to be adjusted if you want to simulate e.g. a hox domain
    #occupying a part of the grid only, which furthermore may change over time
    #Similarly, to simulate a fgf gradient the maximum value of 1 should only
    #be applied to the source grid points
    hox = np.ones((nx, ny))
    fgf = np.ones((nx, ny)) 

    # Simulate one step of reaction and diffusion
    sox9, bmp, wnt, fgf = single_time_step(sox9, bmp, wnt, fgf, hox,
                                      Dbmp, Dwnt, Dfgf,
                                      k2, k3, k4, k5, k7, k9, delta)

    if timestep == 0:  # Initial plot

        current_time = timestep*dt
        
        sox9_plot = ax1.imshow(np.rot90(sox9), cmap=red_dark_cmap, vmin=sox9_vmin, vmax=sox9_vmax)
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot = ax2.imshow(np.rot90(bmp), cmap=green_dark_cmap, vmin=bmp_vmin, vmax=bmp_vmax)
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot = ax3.imshow(np.rot90(wnt), cmap=blue_dark_cmap, vmin=wnt_vmin, vmax=wnt_vmax)
        ax3.set_title(f'wnt at step {current_time}')

        minval = min(np.min(hox), np.min(fgf))
        maxval = max(np.max(hox), np.max(fgf))

        hox_plot = ax4.imshow(np.rot90(hox), cmap='viridis', vmin=np.min(hox), vmax=np.max(hox))
        ax4.set_title(f'hox at step {current_time}')

        fgf_plot = ax5.imshow(np.rot90(fgf), cmap='viridis', vmin=np.min(fgf), vmax=np.max(fgf))
        ax5.set_title(f'fgf at step {current_time}')

        # Use the last axis for a colorbar
        plt.colorbar(fgf_plot, cax=ax6, orientation='horizontal')
        ax6.set_title('hox / fgf colorbar')
        ax6.xaxis.set_ticks_position('bottom')
        ax6.set_position([0.7, 0.2, 0.15, 0.1])
        ax6.yaxis.set_visible(False)

        plt.pause(0.000001)  # Pause to refresh the plot

    elif timestep % 100 == 0:

        current_time = timestep*dt

        sox9_plot.set_data(np.rot90(sox9))    
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot.set_data(np.rot90(bmp))
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot.set_data(np.rot90(wnt))
        ax3.set_title(f'wnt at step {current_time}')

        hox_plot = ax4.imshow(np.rot90(hox), cmap='viridis', vmin=np.min(hox), vmax=np.max(hox))
        ax4.set_title(f'hox at step {current_time}')

        fgf_plot = ax5.imshow(np.rot90(fgf), cmap='viridis', vmin=np.min(fgf), vmax=np.max(fgf))
        ax5.set_title(f'fgf at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot

plt.show()
