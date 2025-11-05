#!/usr/bin/env python3

"""
This module defines the CPM class, the major workhorse of the code-base.

The code was forked from the CPM code of Bao et al., Nature Cell Biology 2022.
GitHub repository: https://github.com/jakesorel/CPM_ETX_2022

The code was changed to improve performance and to remove unneeded parts.
"""

# Installed modules
import numpy as np
from scipy.ndimage import binary_dilation

# Local modules
from .sample import Sample

####################
# Class Definition #
####################

class CPM:
    """
    The CPM class.

    Defines the geometries of cells and their corresponding contacts, then performs Metropolis-Hastings optimisation of
    an energy functional to establish the low-energy conformations of the in-silico tissues.

    """
    def __init__(self, params=None):
        """
        Initialisation of the CPM class.
        @param params: dictionary of parameters
        """
        assert params is not None, "Specify params"
        self.params = params

        # Defined in make_grid()
        self.num_x, self.num_y = None, None
        self.sigma_field   = None
        
        # Defined in define_neighbourhood()
        self.Moore, self.perim_neighbour = None, None

        # Defined in generate_cells()
        self.n_cells = None
        self.c_types = None
        self.cell_ids = None

        # Defined in set_cell_params()
        self.A0, self.P0 = None, None
        self.lambda_P, self.lambda_A = None, None

        # Calculated in assign_AP()
        self.A, self.P = None, None

        # Pre-compute Neumann neighbors for boundary calculations
        self.boundary_calc_ns = np.array([
                                            [ 0, -1],
                                   [-1,  0],         [ 1,  0],
                                            [ 0,  1]
                                ])
        

        # The sample module is the actual simulation engine
        self.sample = Sample(self)

        # Set up the initial configuration
        self.setup_CPM()

    
    def setup_CPM(self):
        """
        Calls all the functions needed to set up the initial configuration.
        """

        self.define_neighbourhood()
        self.make_grid(self.params["nx"], self.params["ny"])
        self.generate_cells(self.params["init_cells"])
        self.set_cell_params()
        self.make_J()
        self.get_J_diff()

        init_type = self.params["init_conf"]
        if init_type == "point":

            # Additional parameters are irrelevant
            self.distribute_cells_on_field(init_type)

        elif init_type == "hex_grid":

            a0 = np.sqrt(max(self.params["A0"]))/ np.pi
            hex_radius = a0 * 0.8 # radius of initial hexagons
            spacing    = a0 * 0.9 # spacing between hexagons

            self.distribute_cells_on_field(init_type, hex_radius, spacing)
        
        self.assign_AP()


    def define_neighbourhood(self):
        """
        Define the multiple neighbourhoods used in calculations.

        Moore neighbourhood (self.Moore) is the 3x3 set of x and y shifts centred on (0,0).

        Perim_neighbour (self.perim_neighbour) is the Moore neighbourhood, minus the central position.
        """
        # Hardcoded Moore neighbourhood
        self.Moore = np.array([
                      [-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0], [ 0,  0], [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])
        
        # Define perimeter neighbourhood by excluding center position
        self.perim_neighbour = np.array([
                      [-1, -1], [ 0, -1], [ 1, -1],
                      [-1,  0],           [ 1,  0],
                      [-1,  1], [ 0,  1], [ 1,  1]])


    def make_grid(self, num_x=300, num_y=300):
        """
        Makes the initial square grid, where the CPM formulation of cellular objects and tissues is housed.

        Initialises sigma_field with zeros (ints).
        sigma_field is a (num_x x num_y) matrix of ints. Each int uniquely corresponds to a cell identity.

        @param num_x: Number of pixels in the x-dimension of the sigma_field.
        @param num_y: Number of pixels in the y-dimension of the sigma_field.
        """
        self.num_x, self.num_y = num_x, num_y
        # Use int32 for better performance on most systems
        self.sigma_field = np.zeros([num_x, num_y], dtype=np.int32)


    def generate_cells(self, cells_to_initialize):
        """
        Initialize a set of cells.
        Defines self.c_types, a list of cell_types, which are indexed 0,1,2,... 
        0 is saved for 'medium' pseudo-cell, whereas 1,2,... are the biological cell types.

        Additionally, defines the number of cells, self.n_cells.
        and the cell_ids, which count up from 1. Again, 0 is reserved for the 'medium' pseudo-cell.

        @param cells_to_initialize: A list of integer numbers corresponding to each cell type.
        e.g. if cells_to_initialize = [8, 4, 6]; there will be 8 of c_type 1; 4 of c_type 2, etc.
        """
        
        c_types_list = []
        for type_i, n_i in enumerate(cells_to_initialize):
            c_types_list.extend([type_i + 1] * n_i)
        
        self.c_types = np.array(c_types_list, dtype=np.int32)
        self.n_cells = len(self.c_types)
        self.cell_ids = np.arange(1, self.n_cells + 1, dtype=np.int32)


    def set_cell_params(self):
        """
        Converts the parameters of the energy functional, prescribed in the dictionary self.params, into vectors of size
        (n_cells +1). N.b. +1, as considers also the medium pseudo-cell.

        Calls the function make_J at the end.

        e.g. if self.params["A0"] = (5,6,7), then self.A0 will be 5, for cell-type 1, will be 6 for cell-type 2 etc.
        """
        # Pre-allocate arrays
        self.A0       = np.zeros(self.n_cells + 1, dtype=np.float32)
        self.P0       = np.zeros(self.n_cells + 1, dtype=np.float32)
        self.lambda_A = np.zeros(self.n_cells + 1, dtype=np.float32)
        self.lambda_P = np.zeros(self.n_cells + 1, dtype=np.float32)

        # Vectorized assignment
        cell_indices = np.arange(1, self.n_cells + 1)
        type_indices = self.c_types - 1
        
        self.A0[cell_indices] = np.array(self.params["A0"])[type_indices]
        self.P0[cell_indices] = np.array(self.params["P0"])[type_indices]
        self.lambda_A[cell_indices] = np.array(self.params["lambda_A"])[type_indices]
        self.lambda_P[cell_indices] = np.array(self.params["lambda_P"])[type_indices]


    def make_J(self):
        """
        In the case of homogeneous prescription of interfacial energies, utilise the "W" matrix.

        W matrix is a (n_cell_type + 1, n_cell_type + 1) of affinities (more positive = stronger affinity).

        This is derived from the self.params dictionary.

        self.J then is a (n_cell+1 x n_cell+1) matrix of interfacial energy coefficients.
        (more negative = stronger affinity).
        n_cell+1 to account for the medium "cell" (ID 0)

        Additionally calls the function get_J_diff.
        """
        self.J = np.zeros([self.n_cells + 1, self.n_cells + 1])
        c_types = np.concatenate(((0,), self.c_types))
        for i in range(len(c_types)):
            for j in range(len(c_types)):
                if i != j:
                    self.J[i, j] = -self.params["W"][c_types[i], c_types[j]]


    def get_J_diff(self):
        """
        J_diff: Change in the interfacial energy when a pixel is replaced from cell index i to cell index j.

        Jdiff is a (nc x nc x nc) array, where the first two dimensions are indices of cells i and j, 
        and the third dimension can be used to index all neighbouring cells of the pixel that is being flipped.
        """
        self.J_diff = self.J[:, np.newaxis, :] - self.J[np.newaxis, :, :]
        

    def distribute_cells_on_field(self, init_type="hex_grid", r=3, spacing=0.25):
        """
        Initialise the sigma_field matrix with cells 
        (i.e. regions of the sigma_field matrix with a given cell index).

        For init_type "point" cells are initialized as a single grid site.

        For init_type "hex_grid" cells are prescribed as circles placed on a hexagonal grid. 
        This is achieved by defining a set of circle centres that are
        grid-tiled in the sigma_field matrix. Cell indices and cell centres are shuffled, 
        such that the initial distribution of cells is random. 

        Extra cells are added, and then are removed from the centre of the
        sigma_field matrix outward until the appropriate number of cells, 
        prescribed in **generate_cells**, is achieved.

        @param init_type: choose the type of initial cell shape, "point" or "circle_grid"
        @param r: Radius of the circle that prescribes the initial grid.
        @param spacing: spacing between the circles.
        """

        sigma_temp = np.zeros_like(self.sigma_field)

        if init_type == "point":
            # Flatten the grid and get all possible indices
            flat_indices = np.arange(self.num_x * self.num_y)
            np.random.shuffle(flat_indices)  # Shuffle the indices

            # Pick the first n_cells indices and assign cell IDs
            chosen_indices = flat_indices[:self.n_cells]

            # Convert flat indices back to 2D coordinates
            y_coords, x_coords = np.unravel_index(chosen_indices, (self.num_x, self.num_y))

            # Assign each cell to a unique location
            for cell_id, (x, y) in enumerate(zip(x_coords, y_coords), start=1):
                sigma_temp[y, x] = cell_id

        elif init_type == "hex_grid":
            
            # Calculate the approximate number of cells along the x and y axes
            # Offset in the x-direction makes it a hexagonal grid
            sq_n_x = int(np.ceil(np.sqrt(self.n_cells))) - 1
            sq_n_y = int(np.ceil(np.sqrt(self.n_cells)) / 2)

            x_mid, y_mid = self.num_x // 2, self.num_y // 2  # center coordinates

            grid_spacing_x = np.arange(-sq_n_x, sq_n_x + 1) * (r + spacing)
            grid_spacing_y = np.arange(-sq_n_y, sq_n_y + 1) * (r + spacing) * np.sqrt(2)

            # Define positions of the cell centers in a grid
            x0s, y0s = x_mid + grid_spacing_x, y_mid + grid_spacing_y

            X0, Y0 = np.meshgrid(x0s, y0s, indexing="ij")
            X0_flat = np.concatenate([X0[::2, ::2].ravel(), X0[1::2, 1::2].ravel()])
            Y0_flat = np.concatenate([Y0[::2, ::2].ravel(), Y0[1::2, 1::2].ravel()])

            # Add small amount of noise to circle centers 
            X0_flat += np.random.uniform(-1, 1, X0_flat.shape)
            Y0_flat += np.random.uniform(-1, 1, Y0_flat.shape)

            # Pick points for all cells based on distance from center of coordinate space
            dist_to_mid = (X0_flat - x_mid) ** 2 + (Y0_flat - y_mid) ** 2
            grid_choice = np.random.permutation(np.argsort(dist_to_mid)[0:self.n_cells+1])

            # Find unique A0 values and their ranges (skipping the 0th element)
            unique_A0 = np.unique(self.A0[1:])  # Skip medium pseudo-cell
            
            # Process cells by A0 value (which corresponds to cell types)
            current_idx = 1  # Start from index 1 (skip medium)
            
            for A0_val in unique_A0:
                
                # Pre-compute circular mask template for this radius
                radius = np.sqrt(A0_val / np.pi) * 0.8
                max_r_int = int(np.ceil(radius)) + 1
                y_template, x_template = np.mgrid[-max_r_int:max_r_int+1, -max_r_int:max_r_int+1]
                circle_mask = x_template**2 + y_template**2 <= radius**2
                
                # Get the relative coordinates where the mask is True
                mask_coords = np.where(circle_mask)
                rel_x, rel_y = mask_coords[0] - max_r_int, mask_coords[1] - max_r_int
                
                # Count how many cells have this A0 value
                cells_with_A0 = np.sum(self.A0[1:] == A0_val)
                cell_range = range(current_idx, current_idx + cells_with_A0)

                # Apply this mask to all cells with this A0 value
                for k in cell_range:
                    grid_k = k - 1  # Convert to 0-based indexing for grid_choice
                    x0, y0 = X0_flat[grid_choice[grid_k]], Y0_flat[grid_choice[grid_k]]
                    
                    # Calculate absolute coordinates by shifting the template
                    abs_x = rel_x + int(x0)
                    abs_y = rel_y + int(y0)
                    
                    # Filter coordinates to stay within bounds
                    valid_coords = ((abs_x >= 0) & (abs_x < self.num_x) & 
                                   (abs_y >= 0) & (abs_y < self.num_y))
                    
                    if np.any(valid_coords):
                        valid_x = abs_x[valid_coords]
                        valid_y = abs_y[valid_coords]
                        sigma_temp[valid_x, valid_y] = k
                
                # Move to the next group of cells
                current_idx += cells_with_A0
            
        self.sigma_field = sigma_temp.copy()
        

    def assign_AP(self):
        """
        Calculate the area and perimeter of every cell in the matrix sigma_field.

        self.A and self.P are (n_cells + 1) vectors of area and perimeter values, respectively. 
        These are indexed via the indices of cells in sigma_field. 
        The first value is that of the medium, which is essentially ignored throughout, 
        as we do not consider the area and perimeter of the medium pseudo-cell in the energy functional.
        """
        self.A = np.zeros(self.n_cells + 1, dtype=np.int32)
        self.P = np.zeros(self.n_cells + 1, dtype=np.int32)
        
        unique_cells, counts = np.unique(self.sigma_field, return_counts=True)
        for cell_id, area in zip(unique_cells, counts):
            if 0 < cell_id <= self.n_cells:
                self.A[cell_id] = area
                self.P[cell_id] = self.get_perimeter(self.sigma_field, cell_id)


    def get_perimeter(self, sigma_field, s):
        """
        Perimeter calculation using scipy operations.
        @param sigma_field: The (num_x x num_y) matrix of ints where each unique entry is a cell ID.
        @param s: Index of the pixel in question.
        """
        # Create binary mask for cell
        mask = (sigma_field == s)
        
        # Use binary_dilation for boundary detection
        dilated = binary_dilation(mask, structure=np.ones((3,3), dtype=bool))
        boundary = dilated & (~mask)
        
        # Count boundary pixels adjacent to cell
        perimeter = 0
        for di, dj in self.boundary_calc_ns:
            shifted_mask = np.roll(np.roll(mask, di, axis=0), dj, axis=1)
            perimeter += np.sum(shifted_mask & boundary)
        
        return perimeter
    

    def get_perimeter_elements(self, sigma_field):
        """
        Returns the perimeter coordinates. Used in plotting.

        Get a mask of all pixels that are adjacent 
        to pixels of different indices (i.e. the cell boundaries).
        
        @param sigma_field: a (num_x x num_y) matrix of ints.

        @return: boundaries, a (num_x x num_y) matrix of ints
                             with 1s corresponding to perimeter pixels
                             and 0 elsewhere.
        """
        # Pre-allocate result array
        boundaries = np.zeros_like(sigma_field, dtype=bool)
        
        # Vectorized boundary detection
        for i, j in self.boundary_calc_ns:
            shifted = np.roll(np.roll(sigma_field, i, axis=0), j, axis=1)
            boundaries |= (sigma_field != shifted) # bitwise OR operation
        
        return boundaries
    

    def initialize(self, J0, n_initialise_steps=10000):
        """
        Initialise the sigma_field matrix by performing M-H steps, 
        after defining the approximate initialisation in **make_init**.

        @param J0: Definition of the J-matrix for the initialisation steps.
        @param n_initialise_steps: Number of initialisation steps.
        """

        if n_initialise_steps > 0:
            print("\nInitializing...")

            # Store original values
            J_orig = self.J.copy()
            lambda_P_orig = self.lambda_P.copy()
            
            # Set initialization parameters
            self.lambda_P[:] = np.max(self.lambda_P)
            self.J.fill(0)
            self.J[1:, 1:] = J0
            np.fill_diagonal(self.J, 0) # Set diagonal elements to 0
            
            self.get_J_diff()

            self.sample.n_steps = n_initialise_steps
            self.sample.do_steps()
            
            # Restore original values
            self.J = J_orig
            self.lambda_P = lambda_P_orig
            self.get_J_diff()

            print("Done with initialisation.\n")


    def simulate(self, updateMode = "traditional"):
        """
        Simulate the CPM algorithm with parameters prescribed in self.params.
        Relies heavily on the **Sample** class in the **sample** module.        
        """

        # Original Monte Carlo algorithm
        if updateMode == "traditional":
            self.sample.n_steps = self.sigma_field.size
            self.sample.monte_carlo_step()
        
        elif updateMode == "edgeList":
            self.sample.n_steps = self.sigma_field.size
            self.sample.monte_carlo_step()
