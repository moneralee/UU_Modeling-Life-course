"""
Implements a simple Cellular Potts Model (CPM).
"""

# System modules
import time

# Installed modules
import numpy as np
import colorcet as cc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#############
# FUNCTIONS #
#############

def get_neighbors(x, y, Lx, Ly, neighbourhood):
    """Get neighbours with periodic boundary conditions"""

    if neighbourhood == "4":
        # Nearest neighbours in a 3x3 cross around x,y
        return [
            (x, (y + 1) % Ly),  # UP
            ((x - 1) % Lx, y),  # LEFT
            (x, (y - 1) % Ly),  # DOWN
            ((x + 1) % Lx, y)   # RIGHT
        ]
    elif neighbourhood == "8":
        # Nearest neighbours in a 3x3 square around x,y
        return [
            (x, (y + 1) % Ly),            # UP
            ((x - 1) % Lx, (y + 1) % Ly), # LEFT-UP
            ((x - 1) % Lx, y),            # LEFT
            ((x - 1) % Lx, (y - 1) % Ly), # LEFT-DOWN
            (x, (y - 1) % Ly),            # DOWN
            ((x + 1) % Lx, (y - 1) % Ly), # RIGHT-DOWN
            ((x + 1) % Lx, y),            # RIGHT
            ((x + 1) % Lx, (y + 1) % Ly)  # RIGHT-UP
        ]
    else: # default to 4-neighbourhood
        print("Invalid neighbourhood chosen. Defaulting to 4-neighbourhood.")
        return [
            (x, (y + 1) % Ly),  # UP
            ((x - 1) % Lx, y),  # LEFT
            (x, (y - 1) % Ly),  # DOWN
            ((x + 1) % Lx, y)   # RIGHT
        ]

def initialize(n_cells, grid_height, grid_width, simparams):
    """
    n_cells has the structure: [n_type1, n_type2, ...]
    """
    
    sigma_field = np.zeros((grid_height, grid_width), dtype=int)
    tau_field   = np.zeros((grid_height, grid_width), dtype=int)

    mode = simparams["init_mode"]

    if mode == "random_pixel":
        """
        Distribute cells at random locations on the grid as 1-pixel points.
        """
        total_cell_idx = 1

        for cell_type_idx, cell_type in enumerate(n_cells):
            for cell in range(cell_type):

                placed = False

                while not placed:
                    # Randomly place the cell on the grid
                    x, y = np.random.randint(0, grid_height), np.random.randint(0, grid_width)

                    # Check if grid place occupied
                    if sigma_field[x, y] != 0:
                        continue # reroll
                    else:
                        placed = True
                        tau_field[x, y]   = cell_type_idx + 1
                        sigma_field[x, y] = total_cell_idx
                        total_cell_idx += 1

    elif mode == "square_grid":
        """Distribute cells evenly spaced across the grid in a square pattern."""

        # Total number of cells
        total_cells = sum(n_cells)

        # Estimate how many grid points per side
        n_per_side = int(np.ceil(np.sqrt(total_cells)))
        size_per_cell = 5

        # Compute offset so grid is centered
        x_center = grid_height // 2 - (n_per_side*size_per_cell) // 2 
        y_center = grid_width  // 2 - (n_per_side*size_per_cell) // 2 

        # Generate grid positions (centers of each cell)
        cell_positions = []
        for i in range(n_per_side):
            for j in range(n_per_side):
                x = int((i + 0.5) * size_per_cell) + x_center
                y = int((j + 0.5) * size_per_cell) + y_center
                if x < grid_height and y < grid_width:
                    cell_positions.append((x, y))

        # Remove unneeded positions then shuffle
        pos2remove = len(cell_positions) - total_cells
        if pos2remove % 2 == 0:
            cell_positions = cell_positions[pos2remove//2 : -pos2remove//2]
        else:
            cell_positions = cell_positions[pos2remove//2 : -(pos2remove//2 + 1)]
        np.random.shuffle(cell_positions)

        # Assign cells to positions
        total_cell_idx = 1
        pos_idx = 0
        for cell_type_idx, cell_type in enumerate(n_cells):
            for cell in range(cell_type):
                x, y = cell_positions[pos_idx]

                # Mark center pixel
                sigma_field[x, y] = total_cell_idx
                tau_field[x, y]   = cell_type_idx + 1

                # Fill in neighbors (square patch)
                neighbourhood = get_neighbors(
                    x, y, sigma_field.shape[0], sigma_field.shape[1],
                    neighbourhood="8"
                )
                for nx, ny in neighbourhood:
                    sigma_field[nx, ny] = total_cell_idx
                    tau_field[nx, ny]   = cell_type_idx + 1

                total_cell_idx += 1
                pos_idx += 1

    else:
        raise ValueError(f"Unknown mode '{mode}'")

    return(sigma_field, tau_field)


def get_dH_adhesion(x, y, target_sigma, 
                    nx, ny, source_sigma, 
                    sigma_field, tau_field, 
                    sigma_copy, tau_copy, 
                    parameters):
    """
    ADHESION CALCULATION
    1. Sum the adhesion energy of the source site to each neighbor before copy.
    2. Sum the adhesion energy of the target site to each neighbor after copy.
    3. Calculate the difference in adhesion energy between current and new configuration.
    """

    # Get the adhesion table from parameters
    adhesion_table = parameters['adhesion_table']

    # Get neighbours of the target
    neighbours = get_neighbors(x, y, sigma_field.shape[0], sigma_field.shape[1], parameters["neighborhood"])
    
    adhesion_before = 0
    adhesion_after = 0

    for nnx, nny in neighbours:
        # Step 1
        # Adhesion energy of target neighbourhood before copy
        neighbor_cell_id = sigma_field[nnx, nny]
        if neighbor_cell_id != target_sigma:
            adhesion_before += adhesion_table[tau_field[x, y], tau_field[nnx, nny]] 

        # Step 2
        # Adhesion energy of target neighbourhood after copy
        neighbor_cell_id = sigma_copy[nnx, nny]
        if neighbor_cell_id != source_sigma:
            adhesion_after += adhesion_table[tau_copy[x, y], tau_copy[nnx, nny]]  
    
    # Step 3
    # Difference in adhesion energy
    dH = adhesion_after - adhesion_before

    return(dH)


def get_dH_volume(x, y, target_sigma, 
                  nx, ny, source_sigma, 
                  sigma_field, tau_field, 
                  sigma_copy, tau_copy, 
                  parameters):
    """
    VOLUME CALCULATION
    1. For both cell types involved in copy attempt check first 
       if they are actual cells and not medium (tau > 0).
       If yes, calculate the volume before and after the copy attempt.
    2. Calculate the energy contribution from volume constraint before and after.
    """

    # Get volume parameters
    volume_ideal  = parameters['volume_ideal']
    volume_weight = parameters['volume_weight']

    # Step 1
    target_cell = tau_field[x, y]   # cell type index of target cell
    source_cell = tau_field[nx, ny] # cell type index of source cell

    # Initialize variables
    vol_target_energy_before = 0
    vol_target_energy_after  = 0
    vol_source_energy_before = 0
    vol_source_energy_after  = 0

    if target_cell != 0:
        volume_target_before = np.sum(sigma_field == target_sigma)
        volume_target_after  = np.sum(sigma_copy == target_sigma)

        vol_target_energy_before = volume_weight[target_cell - 1] * (volume_target_before - volume_ideal[target_cell - 1])**2
        vol_target_energy_after  = volume_weight[target_cell - 1] * (volume_target_after - volume_ideal[target_cell - 1])**2
    
    if source_cell != 0:
        volume_source_before = np.sum(sigma_field == source_sigma)
        volume_source_after  = np.sum(sigma_copy == source_sigma)

        vol_source_energy_before = volume_weight[source_cell - 1] * (volume_source_before - volume_ideal[source_cell - 1])**2
        vol_source_energy_after  = volume_weight[source_cell - 1] * (volume_source_after - volume_ideal[source_cell - 1])**2

    # Step 2
    volume_energy_before = vol_target_energy_before + vol_source_energy_before
    volume_energy_after  = vol_target_energy_after  + vol_source_energy_after

    dH = volume_energy_after - volume_energy_before

    return(dH)


def calc_dH(x, y, target_sigma, 
            nx, ny, source_sigma, 
            sigma_field, tau_field, 
            sigma_copy, tau_copy, 
            parameters):
    """
    Calculate the change in energy dH if a copy were to happen.
    """

    dH = 0

    dH += get_dH_adhesion(x, y, target_sigma, 
                          nx, ny, source_sigma, 
                          sigma_field, tau_field, 
                          sigma_copy, tau_copy, 
                          parameters)
    dH += get_dH_volume(x, y, target_sigma, 
                        nx, ny, source_sigma, 
                        sigma_field, tau_field, 
                        sigma_copy, tau_copy, 
                        parameters)
    return(dH)


def monte_carlo_step(sigma_field, tau_field, parameters):
    """
    Modified Metropolis algorithm.
    Do a number of copy attempts:
    1. Pick a random grid space x_T <-- "target pixel"
    2. Pick a random neighbor x_S of that grid space <-- "source pixel"
    3. Calculate the change in energy dH between new and old spatial configurations. 
       The new configuration would result if the value of the source pixel 
       would be copied to the target pixel.
    4. Accept or reject the copy according to the Metropolis criterion
       (always accept if dH < 0, otherwise accept with probability exp(-dH/kT))
    """

    accepted_copies = 0
    rejected_copies = 0
    total_attempts  = sigma_field.size

    for copy_attempt in range(total_attempts):
        # 1. Pick a random grid space x_T <-- "target pixel"
        x, y = np.random.randint(0, sigma_field.shape[0]), np.random.randint(0, sigma_field.shape[1])
        target_sigma = sigma_field[x, y]

        # 2. Pick a random neighbor x_S of that grid space <-- "source pixel"
        neighbors = get_neighbors(x, y, sigma_field.shape[0], sigma_field.shape[1], parameters["neighborhood"])
        nx, ny = neighbors[np.random.randint(len(neighbors))]
        source_sigma = sigma_field[nx, ny]
        source_tau   = tau_field[nx, ny]

        rejected = False

        # Always reject if source and target are the same cell ID
        if source_sigma == target_sigma:
            rejected_copies += 1
            rejected = True
        
        # Make a copy of the sigma and tau fields, assuming the copy attempt succeeds
        sigma_copy = sigma_field.copy()
        sigma_copy[x, y] = source_sigma

        tau_copy = tau_field.copy()
        tau_copy[x, y] = source_tau

        if not rejected:
            # 3. Calculate the change in energy dH if a copy were to happen
            dH = calc_dH(x, y, target_sigma, 
                         nx, ny, source_sigma, 
                         sigma_field, tau_field, 
                         sigma_copy, tau_copy, 
                         parameters)
            
            # 4. Accept or reject the copy according to the Metropolis criterion
            # accept
            if dH < 0 or np.random.rand() < np.exp(-dH / parameters['T']):
                sigma_field[x, y] = source_sigma
                tau_field[x, y]   = source_tau
                accepted_copies += 1
            # reject
            else:
                rejected_copies += 1

    return(accepted_copies, rejected_copies)

############
# Plotting #
############

def init_plot(sigma_field, tau_field, zero_color='white'):

    # Create a colormap: first color is zero_color, rest are from "glasbey" colormap
    glasbey_colors = list(cc.glasbey)
    cmap_colors = [zero_color] + glasbey_colors
    cmap = ListedColormap(cmap_colors)

    ax1, ax2 = plt.subplots(1, 2, figsize=(8, 4))[1].flatten()

    # First two subplots for sigma and tau fields
    plot1 = ax1.imshow(sigma_field, cmap=cmap, vmin=0, vmax=len(cmap_colors)-1)
    ax1.set_title(f"Individual Cells (sigma_field)")
    ax1.axis('off')
    plot2 = ax2.imshow(tau_field, cmap=cmap, vmin=0, vmax=len(cmap_colors)-1)
    ax2.set_title(f"Cell Types (tau_field)")
    ax2.axis('off')

    plt.tight_layout()
    plt.pause(0.000001)

    return(plot1, plot2)

def update_plots(sigma_field, tau_field, plot1, plot2):
    plot1.set_data(sigma_field)
    plot2.set_data(tau_field)

    plt.pause(0.000001)


##################
# EXECUTION CODE #
##################
if __name__ == "__main__":
    
    ##############
    # PARAMETERS #
    ##############

    # Grid size
    height = 50 # number of rows
    width  = 50 # number of cols

    initial_cell_number = [10, 10]

    total_steps = 100

    """
    The following parameters are the default values used in:
    Graner, François, and James A. Glazier. 
    "Simulation of biological cell sorting using a 
    two-dimensional extended Potts model." 
    Physical review letters 69.13 (1992): 2013.
    """

    # "Temperature" of the simulation
    # A measure for how much energy the cells are willing to spend
    # to make a physically unfavourable move
    temperature = 10

    # Define adhesion penalties between different cell types
    j_01 = 16  # adhesion cell type 1 - medium
    j_02 = 16  # adhesion cell type 2 - medium
    j_11 = 2   # adhesion cell type 1 - cell type 1
    j_12 = 11  # adhesion cell type 1 - cell type 2
    j_22 = 14  # adhesion cell type 2 - cell type 2

    adhesion_table = np.array([
    [    0,  j_01, j_02], # adhesion medium - partner
    [  j_01, j_11, j_12], # adhesion cell type 1 - partner
    [  j_02, j_12, j_22], # adhesion cell type 2 - partner
    ])
    
    cell_volume = [40, 40]

    # For convenience, pack all parameters into a dictionary
    sim_parameters = {
        'init_mode'      : 'random_pixel',
        'neighborhood'   : "4",
        'T'              : temperature,
        'adhesion_table' : adhesion_table,

        # Volume of the cell (area in 2D)
        'volume_ideal'   : cell_volume,
        'volume_weight'  : [1, 1]
    }

    ##############
    # SIMULATION #
    ##############

    # Initialize the simulatio and plots
    sigma_field, tau_field = initialize(initial_cell_number, height, width, sim_parameters)
    plot1, plot2 = init_plot(sigma_field, tau_field)

    # Create lists to track copy attempts
    accepted_copies = []
    rejected_copies = []

    # Run the simulation
    start_time = time.time() # To measure the run time

    for sim_step in range(int(total_steps)):
        accepted, rejected = monte_carlo_step(sigma_field, tau_field, sim_parameters)
        
        accepted_copies.append(accepted)
        rejected_copies.append(rejected)
        
        update_plots(sigma_field, tau_field, plot1, plot2)

    end_time = time.time()   # To measure the run time

    # Print data to terminal
    for mcs, (accept, reject) in enumerate(zip(accepted_copies, rejected_copies)):
        print("Monte Carlo Step", mcs+1, 
              "\tAccepted", accept, 
              "\tRejected", reject, 
              "    Total", accept+reject)

    print()
    print(f"Execution time: {end_time - start_time:.2f} seconds")

    plt.show()