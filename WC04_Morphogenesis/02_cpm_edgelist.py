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


def initialize_edgelist(sigma_field, parameters):
    """
    validEdges is a "flat" 1-dimensional representation of all pairs of lattice sites 
    (including the boundary, even in the case of impassable boundary).
    The list validEdges is filled with unique non-negative values for each pair 
    of lattice sites with unequal non-negative spins.

    The list validEdgeIndices stores the indices of the unique 
    non-negative values in validEdges. It is bijective with validEdges.

    The validIndicesShort is equivalent to validEdges without negative values.
    It facilitates looping in the monte carlo step.

    Algorithm:
    1. Initialize validEdges with size corresponding to number of 
        neighbour pairs in the lattice, filled with -1.
        Initialize two counters:
        - the index of the current edge current edge "edge_index" with 0
        - the unique edge integer value "index_edgelist" with 0
    2. Loop over the lattice and the neighbours.
    3. For neighbouring lattice sites of unlike sigma >=0 (i.e. ignore border), 
        - validEdges: add the value index_edgelist at the index edge_index 
        - validEdgeIndices: add the value edge_index at the index index_edgelist 
        - increment index_edgelist
    4. At the end of each loop iteration, increment edge_index.
    5. Store the total number of valid edges.
    """

    height, width = sigma_field.shape
    neighbours0 = get_neighbors(0, 0, height, width, parameters["neighborhood"])

    # step 1 -- NOTE: -1 is the state of "no flips can happen"
    validEdges = np.full(width * height * len(neighbours0), -1)
    validEdgeIndices = np.full(width * height * len(neighbours0), -1)

    # step 2
    edge_index = 0
    index_edgelist = 0
    
    for x in range(height):
        for y in range(width):
            sigma = sigma_field[x, y]
            neighbours = get_neighbors(x, y, height, width, parameters["neighborhood"])

            # Check all neighbours
            for nx, ny in neighbours:
                neighbor_sigma = sigma_field[nx, ny]

                # step 3 -- Add edge if neighboring sigmas are different and both valid
                if sigma != neighbor_sigma and sigma >= 0 and neighbor_sigma >= 0:
                    validEdges[edge_index] = index_edgelist
                    validEdgeIndices[index_edgelist] = edge_index

                    index_edgelist += 1
                
                # step 4
                edge_index += 1

    # step 5
    totalValidEdges = index_edgelist
    validIndicesShort = validEdgeIndices[0:totalValidEdges]

    return(validEdges, validEdgeIndices, validIndicesShort, totalValidEdges)


def mapEdgeIndexToCoordinate(edge_index, width, parameters):
    """ 
    Maps the index of the edge in the validEdges to 
    the x,y coordinate where this edge originates and
    the neighbor index n.
    
    Because the loop that builds the validEdges goes over x first,
    only the width is needed to map back.
    """

    neighbours = get_neighbors(0,0, width, width, parameters["neighborhood"])

    # Calculate the neighbor index
    n = edge_index % len(neighbours)

    # Calculate the lattice point index
    lattice_point_index = edge_index // len(neighbours)

    # Calculate the (x, y) coordinates
    # divmod is a built-in python function that does the same as:
    # x = lattice_point_index // width
    # y = lattice_point_index % width
    x, y = divmod(lattice_point_index, width)

    return([(x, y), n])


def mapCoordinateToEdgeIndex(x, y, neighbor_index, height, width, parameters):
    """ outward_edges = ( x * width + y )*len(NEIGHBORS) + n """
    
    neighbours = get_neighbors(x, y, height, width, parameters["neighborhood"])
    edge_index = ( x * width + y )*len(neighbours) + neighbor_index

    return(edge_index)


def updateEdgelistOnCopy(x, y, sigma_field, parameters,
                         validEdges, validEdgeIndices, totalValidEdges):
    """
    All edges of the target site must be checked to see if they
    need to be updated.

    The number of edges to be updated is 2*len(NEIGHBORS), 
    half being outward edges and half being inward edges.

    To update the outward edges, just find all outward edges
    coming from the target site.
    To update all inward edges, check the edges of the neighbors 
    of the target site.

    NOTE: to function properly this function needs to be called 
    after the sigma field has been updated. Otherwise the old
    sigma value will be checked and none of the edges will update.
    """

    def getCounterEdgeIndex(n, neighborhood):
        """
        Return the counter edge index of the neighbor n.
        The indices are shifted by half of the length of
        the number of neighbors. 
        NOTE: Works for even number of neighbors only!
        """

        focal_indices   = list(range(neighborhood))
        counter_indices = focal_indices[neighborhood//2:] + focal_indices[0:neighborhood//2]

        return(counter_indices[n])
    
    height, width = sigma_field.shape
    target_sigma = sigma_field[x, y]

    # Get neighbours of the target site
    neighbors = get_neighbors(x, y, height, width, parameters["neighborhood"])

    netEdgesChanged = 0
    
    # Check each neighbor to see if edge/counteredge pair needs updating
    for n, (nx, ny) in enumerate(neighbors):

        # neighbor sigma
        nSigma  = sigma_field[nx, ny]

        # edge and counteredge
        edge             = mapCoordinateToEdgeIndex(nx, ny, n, height, width, parameters)
        counterEdgeIndex = getCounterEdgeIndex(n, len(neighbors))
        counterEdge      = mapCoordinateToEdgeIndex(nx, ny, counterEdgeIndex, height, width, parameters)

        # unequal sigma but not in validEdges --> add edge and counteredge
        if target_sigma != nSigma and validEdges[edge] == -1:

            # add edge
            validEdges[edge] = totalValidEdges
            validEdgeIndices[totalValidEdges] = edge
            totalValidEdges += 1

            # add counteredge
            validEdges[counterEdge] = totalValidEdges
            validEdgeIndices[totalValidEdges] = counterEdge
            totalValidEdges += 1

            netEdgesChanged += 2

        # equal sigma but in validEdges --> remove edge and counteredge
        if target_sigma == nSigma and validEdges[edge] > -1:

            # remove edge
            lastValidIndex = totalValidEdges - 1
            
            # check if the edge to be removed is the last one in the list
            if validEdges[edge] != lastValidIndex:
                # replace edge position with last valid index so that
                # validEdgeIndices remains a consecutive list of valid edges 
                validEdgeIndices[validEdges[edge]] = validEdgeIndices[lastValidIndex]

                # update validEdges to point to the new index
                validEdges[validEdgeIndices[lastValidIndex]] = validEdges[edge] 
            
            validEdges[edge] = -1                  # reset value
            validEdgeIndices[lastValidIndex] = -1  # remove last entry
            totalValidEdges -= 1                   # decrement total valid edges
            
            # remove counteredge in same way
            lastValidIndex = totalValidEdges - 1
            
            if validEdges[counterEdge] != lastValidIndex:
                validEdgeIndices[validEdges[counterEdge]] = validEdgeIndices[lastValidIndex]
                validEdges[validEdgeIndices[lastValidIndex]] = validEdges[counterEdge] 
            
            validEdges[counterEdge] = -1           
            validEdgeIndices[lastValidIndex] = -1  
            totalValidEdges -= 1                   
            
            netEdgesChanged -= 2

    # Update the short list
    validIndicesShort = validEdgeIndices[0:totalValidEdges]

    return(validEdges, validEdgeIndices, validIndicesShort, totalValidEdges, netEdgesChanged)


def monte_carlo_step_edgelist(sigma_field, tau_field, parameters, 
                              validEdges, validEdgeIndices, validIndicesShort, totalValidEdges):
    """
    Do a number of copy attempts equal to the number of grid spaces:
    1. Pick a random edge e
    1a. Find the target site belonging to this edge
    1b. Find the source site belonging to this edge
    2. Calculate the change in energy dH if the value of the source pixel
       would be copied to the target pixel
    3. Accept or reject the copy according to the Metropolis criterion
       (always accept if dH < 0, otherwise accept with probability exp(-dH/kT))
    """

    accepted_copies = 0
    rejected_copies = 0
    n_nb = len(get_neighbors(0,0, sigma_field.shape[0], sigma_field.shape[1], parameters["neighborhood"]))
    total_attempts  = totalValidEdges/n_nb

    copy_attempts = 0

    while copy_attempts < total_attempts:
        # 1. Pick a random edge e
        random_edge_index = np.random.choice(validIndicesShort)
        
        # 1a. Find the target site belonging to this edge
        (x, y), n = mapEdgeIndexToCoordinate(random_edge_index, sigma_field.shape[0], parameters)
        target_sigma = sigma_field[x, y]

        # 1b. Find the source site belonging to this edge
        neighbors = get_neighbors(x, y, sigma_field.shape[0], sigma_field.shape[1], parameters["neighborhood"])
        nx, ny = neighbors[n]
        source_sigma = sigma_field[nx, ny]
        source_tau   = tau_field[nx, ny]

        rejected = False
        
        # Make a copy of the sigma field, assuming the copy attempt succeeds
        sigma_copy = sigma_field.copy()
        sigma_copy[x, y] = source_sigma

        tau_copy = tau_field.copy()
        tau_copy[x, y] = source_tau

        if not rejected:
            # 2. Calculate the change in energy dH if a copy were to happen
            dH = calc_dH(x, y, target_sigma, 
                         nx, ny, source_sigma, 
                         sigma_field, tau_field, 
                         sigma_copy, tau_copy, 
                         parameters)
            
            # 3. Accept or reject the copy according to the Metropolis criterion
            # accept
            if dH < 0 or np.random.rand() < np.exp(-dH / parameters['T']):
                sigma_field[x, y] = source_sigma
                tau_field[x, y]   = source_tau
                accepted_copies += 1
                
                # Update the edgelist after the copy
                updated_edgedata = updateEdgelistOnCopy(x, y, sigma_field, parameters,
                                     validEdges, validEdgeIndices, totalValidEdges)
                
                validEdges        = updated_edgedata[0]
                validEdgeIndices  = updated_edgedata[1]
                validIndicesShort = updated_edgedata[2]
                totalValidEdges   = updated_edgedata[3]
                netEdgesChanged   = updated_edgedata[4]

                # Update the loop variable total_attempts accordingly
                total_attempts += netEdgesChanged / n_nb

            # reject
            else:
                rejected_copies += 1
        
        copy_attempts += 1
    
    return(accepted_copies, rejected_copies, updated_edgedata[0:4])


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

    # Initialize the simulation and plots
    sigma_field, tau_field = initialize(initial_cell_number, height, width, sim_parameters)
    plot1, plot2 = init_plot(sigma_field, tau_field)

    edgedata = initialize_edgelist(sigma_field, sim_parameters)

    # Create lists to track copy attempts
    accepted_copies = []
    rejected_copies = []

    # Run the simulation
    start_time = time.time() # To measure the run time

    for sim_step in range(int(total_steps)):
        accepted, rejected, edgedata = monte_carlo_step_edgelist(sigma_field, tau_field, sim_parameters, *edgedata)
        
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