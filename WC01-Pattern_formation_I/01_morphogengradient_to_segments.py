import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx = 40.0  # Length of the domain in x in microm
Ly = 10.0  # Length of the domain in y in microm
T = 200  # Total time in seconds
dx = 0.5  # Grid spacing in x
dt = 0.1  # Time step
nx = int(Lx/dx)+2  # Number of grid points in x + padding grid points
ny = int(Ly/dx)+2  # Number of grid points in y + padding grid points
# Padding grid points to account for boundary conditions
nt = int(T/dt)  # Number of time steps
D = 0.4  # Diffusion coefficient in mm^2/s
decayM =0.01 # Decay rate in 1/s


# Parameters for A, B, C
... # TODO create parameters for A, B, C as needed in Q5

# Stability criterion
if D * dt / dx**2 > 0.5:
    raise ValueError("Stability criterion not met")

# A, B and C are required for later exercises.
A = np.zeros((nx, ny))
B = np.zeros((nx, ny))
C = np.zeros((nx, ny))

# Initial condition
u = np.zeros((nx, ny))
u[0, :] = 100

# Reaction-diffusion equation
def reaction_diffusion_step(u, D, dt, dx, decay):
    un = u.copy()
    u[1:-1, 1:-1] = un[1:-1, 1:-1] +  D *dt / dx**2 * (un[2:, 1:-1] + un[:-2, 1:-1] + \
                    un[1:-1, 2:]  + un[1:-1, :-2] - 4 * un[1:-1, 1:-1]) - \
                    decay * un[1:-1, 1:-1] * dt
    ## for loop version to understand the equation
    # for i in range(1, nx-1):
    #     for j in range(1, ny-1):
    #         u[i, j] = (un[i, j] +
    #                    D * dt / dx**2 * (un[i+1, j] + un[i-1, j] - 2 * un[i, j] +
    #                    un[i, j+1] + un[i, j-1] - 4 * un[i, j]) - decay * un[i, j] * dt)
    #boundary conditions
    u[-1, :] = (u[-2, :]/u[-3, :])*u[-2, :]  if sum(u[-3, :]) != 0 else np.zeros(ny)
    #to understand this line:
    #if sum(u[-3, :]) != 0:
    #    u[-1, :] = (u[-2, :]/u[-3, :])*u[-2, :]#extrapolate from third to last row
    #else:
    #    u[-1, :] = np.zeros(ny) #if already zero in third to last row, also zero in last row
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]

    return u

def reaction_diffusion_gradient(t, u, D, dx, decay, switch_time = None, noise = False):
    '''
    Function to create a gradient in the u array that could decay after a certain time.
    t: current time step
    u: array to create the gradient in
    D: diffusion coefficient
    dx: grid spacing
    decay: decay rate
    switch_time: time step after which the gradient decays. If no switch is desired, set to None
    noise: whether to add noise to the gradient
    '''
    # TODO for student: write code for the noise and the switch.
    added_noise = np.zeros_like(u)  # Initialize noise array
    if noise:
        ...  # TODO: add noise generation code here for Q10
    
    if switch_time is None or t <= switch_time:
        # define a exponential decay gradient over the array in the x direction with numpy array operations using the index
        for i in range(u.shape[0]):
            u[i, :] = np.maximum(100 * np.exp(-i*dx/np.sqrt(D/decay))+added_noise[i, :], 0)
        return u
    if t > switch_time:
        ...# TODO Q7: implement a gradient that decays over time, otherwise return the original u array
        return u
    # In all other cases, return the original u array        
    return u

def hill(x, Km, pow):
    """Hill function for the reaction kinetics."""
    return (x**pow) / (Km**pow + x**pow) 

def ihill(y, Km, pow):
    """Inverse Hill function for the reaction kinetics."""
    return( (Km**pow) / (y **pow  + Km**pow))

# TODO for student: write update functions for A, B, C as needed in Q5


# initilize figure and axes for plotting
# TODO for student: Add a new axis for the ABC flag visualization as suggested in Q5
fig, (ax_M, ax_lines) = plt.subplots(2, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})  # Make the first graph 3 times the height of the second

# Time-stepping simulation loop
for n in range(nt):
    # Update all variables
    u = reaction_diffusion_step(u, D, dt, dx, decayM)
    # TODO for student: use precomputed gradient, update A, B, C as needed in Q5
    
    if n == 0:  # Initial plot
        imshow_M_plot = ax_M.imshow(u.T, cmap='viridis', origin='lower', aspect='auto')
        ax_M.set_title(f"Time: {n*dt}")
        ax_M.set_xlabel('x direction')
        ax_M.set_ylabel('y direction')
        ax_M.set_xticks([])
        ax_M.set_yticks([])

        # Plot the concentration at a specific y index (e.g., y=2)    
        line_plot = ax_lines.plot([x*dx for x in range(nx)], u[:, 2], label='M', color='green')
        # TODO: Add lines for A, B, C as needed in Q5

        
        ax_lines.legend(loc='upper right')
        ax_lines.set_ylim(0, 100)
        ax_lines.tick_params(axis='y')
        ax_lines.set_xlim(0, dx*nx)
        ax_lines.set_xlabel('x')
        ax_lines.set_ylabel('Concentration at y=2')
        ax_lines.tick_params(axis='x')

    if n % 20 == 0:  # Update plot every so many time steps
        #update the imshow M plot with the new data
        imshow_M_plot.set_data(u.T)
        ax_M.set_title(f"Time: {n*dt}")
            
        # Update the line plots with new data
        line_plot[0].set_ydata(u[:, 2])
        # TODO: Update A, B, C line plots as needed in Q5

    plt.pause(0.001)  # Pause to refresh the plot

# And keep the last plot open
# plt.show()

# Or close the plot window when done
plt.close(fig)