#!/usr/bin/env python3

"""
This module takes in a CPM object, containing information about tissue geometry,
and evolves it under the Cellular-Potts/Metropolis Hastings algorithm to stochastically 
minimize the energy functional.

"""

# Installed modules
import numpy as np
from numba import jit

# Local modules
from .zmasks import Zmasks


####################
# Class Definition #
####################

class Sample:
    """
    Sample class, wrapping functions that perform 
    the Metropolis-Hastings sampling algorithm.
    """

    def __init__(self, cpm, n_steps=100):
        """
        Initialise Sample class.
        @param cpm: a CPM object, on which the Metropolis-Hastings 
                    optimisation algorithm is performed.
                    
        @param n_steps: number of steps to perform, every time 
                        the single_steps function is called.
        """
        self.cpm = cpm          # a CPM object.
        self.zmasks = Zmasks()  # initialise the Zmasks object for connectivity constraint.
        self.primes = self.zmasks.primes
        self.hashes = self.zmasks.hashes

        self.T = self.cpm.params["T"]  # alias for the temperature of the optimisation. Prescribed in the **CPM** class.
        self.n_steps = n_steps  # number of steps to perform, every time the **single_steps** function is called.

        # Pre-compute constants
        self._boundary_margin = 1  # Margin from boundaries
        self._neumann_neighbors = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int32)
        
        # Pre-compute Moore neighborhood indices excluding the center pixel
        self._moore_indices = np.array([0, 1, 2, 3, 5, 6, 7, 8], dtype=np.int32)


    def copy_attempt(self):
        """
        Wrapper for the copy_attempt function, see below.

        Performs the function, then transfers the results back to the CPM class.
        """
        self.cpm.sigma_field, self.cpm.A, self.cpm.P = copy_attempt(self.cpm.sigma_field, self.cpm.num_x, 
                                                               self.cpm.num_y, self.zmasks.dP_z,
                                                               self.cpm.A, self.cpm.P, 
                                                               self.cpm.lambda_A, self.cpm.lambda_P,
                                                               self.cpm.A0, self.cpm.P0, 
                                                               self.cpm.J_diff, self.T,
                                                               self.primes, self.hashes,
                                                               self._neumann_neighbors, self._moore_indices)

    def monte_carlo_step(self):
        """
        Wrapper for the monte_carlo_step function, see below.
        monte_carlo_step is itself a wrapper for the function single_step.

        Performs the function, then transfers the results back to the CPM class.
        """
        self.cpm.sigma_field, self.cpm.A, self.cpm.P = monte_carlo_step(self.n_steps, self.cpm.sigma_field, 
                                                                self.cpm.num_x, self.cpm.num_y,
                                                                self.zmasks.dP_z, self.cpm.A, 
                                                                self.cpm.P, self.cpm.lambda_A,
                                                                self.cpm.lambda_P, self.cpm.A0, 
                                                                self.cpm.P0, self.cpm.J_diff,
                                                                self.T, self.primes, self.hashes,
                                                                self._neumann_neighbors, self._moore_indices)


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def copy_attempt(I, num_x, num_y, 
                dP_z, A, P, lambda_A, lambda_P, A0, P0, 
                J_diff, T, primes, hashes,
                swap_neighborhood, moore_indices):
    """
    Performs a single copy attempt of the Metropolis-Hastings algorithm.

    @param I: Sigma field, a (num_x x num_y) matrix of ints. 
              Each unique int corresponds to a cell.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.
    @param dP_z: The change of perimeter given a specific type of swap. 
                 Indexed with respect to the z-mask list 
                 (see documentation in the **zmasks** module).
    @param A: Vector of areas, indexed with respect to cell indices prescribed in I.
    @param P: Vector of cell perimeters, indexed with respect to 
              cell indices prescribed in I.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.
    @param J_diff: Change in the interfacial energy when 
                   a pixel is replaced from cell index i to cell index j.
                   Jdiff is a (nc x nc x nc) array, where the first two 
                   dimensions are indices of cells i and j, and the third
                   dimension can be used to index of all neighbouring cells 
                   of the pixel that is being flipped.
    @param T: Pseudo-temperature, used in the Metropolis-Hastings algorithm.
    @param primes: The kernel used to hash the local Moore neighbourhood.
    @param hashes: The set of hashes for all accepted local Moore neighbourhoods.

    @return: I_new, the new sigma matrix. A, the new areas. P, the new perimeters.
    """

    # Given an existing I matrix, sample a random point, 
    # and then select the state of one of its Neumann neighbours.
    # (i,j) is the coordinate in the I matrix of the selected pixel.
    # s is the cell index of the chosen pixel
    # s2 is the cell index of the chosen neighbour
    i, j, s, s2 = pick_pixel(I, num_x, num_y, swap_neighborhood)

    if s == s2:
        accept_move = False
        return I, A, P

    # Initialise the changes in the Energy/Hamiltonian as 0.
    dH_1, dH_2 = 0, 0

    # Given a point (i,j), subset I to find the Moore neighbourhood. Na is a matrix of size (3,3)
    Na = get_Na(I, i, j)

    # Initialise mask_ids.
    mask_id_1, mask_id_2 = 0, 0

    # Initialise the contributions of the change in the energy.
    dP_1 = 0
    dP_2 = 0
    dA_1 = 0
    dA_2 = 0

    # Calculate the change in energy. 
    # Calculated as two components: changes in state s and changes in state s2.

    if s != 0:  # if the chosen pixel is not a medium cell.

        # The mask Na==s generically defines the neighborhood. 
        # Only certain masks are allowed in order to preserve
        # local Moore contiguity and hence global Moore contiguity.
        # This is achieved by indexing the hashes of the allowed masks.
        # mask_id is the index of the mask Na==s within the 
        # pre-defined list of acceptable masks.

        mask_id_1 = get_mask_id(Na == s, primes, hashes)

        # if Na==s is in the list of acceptable masks, 
        # then calculate the changes in area and perimeter 
        # and hence the energy change.
        if mask_id_1 != -1:  
            dP_1 = dP_z[mask_id_1]
            dA_1 = -1
            dH_1 = get_dH(s, dP_1, dA_1, A, P, lambda_A, lambda_P, A0, P0)

    if s2 != 0:  # if the chosen neighbor state is not a medium cell.

        mask_id_2 = get_mask_id(Na == s2, primes, hashes)

        if mask_id_2 != -1:
            dP_2 = dP_z[mask_id_2]
            dA_2 = 1
            dH_2 = get_dH(s2, dP_2, dA_2, A, P, lambda_A, lambda_P, A0, P0)
            
    # Calculate the change in the interfacial energy term.
    dJ = get_dJ(J_diff, s, s2, Na, moore_indices)

    # Sum together the changes in the contributions 
    # of the energy to calculate the total change in energy: dH.
    dH = dH_1 + dH_2 + dJ
    
    accept_move = False
    # if both masks (Na==s) and (Na==s2) are permissible.
    if (mask_id_1 != -1) * (mask_id_2 != -1):

        # if the change in energy is less than or equal to 0 --> always accept
        if dH <= 0:  
            accept_move = True

        # change in energy greater than 0 --> accept with Boltzmann probability
        # stochastic contribution to minimisation, under M-H.
        else:
            prob = np.exp(-dH / T)
            if np.random.random() < prob:  
                accept_move = True

    if accept_move:
        # Swap the state of pixel (i,j) with the state of the neighbours.
        I[i, j] = s2

        # Update the properties of the cells.
        A[s]  += dA_1
        A[s2] += dA_2
        P[s]  += dP_1
        P[s2] += dP_2

    return I, A, P


@jit(nopython=True, cache=True)
def monte_carlo_step(n_steps, I, num_x, num_y, 
                dP_z, A, P, lambda_A, lambda_P, A0, P0, 
                J_diff, T, primes, hashes,
                swap_neighborhood, moore_indices):
    """
    Iterate single_step for n_steps.

    @return: I, the new sigma_field matrix. A, the new areas. P, the new perimeters.
    """
    for _ in range(n_steps):
        I, A, P = copy_attempt(I, num_x, num_y, 
                              dP_z, A, P, lambda_A, lambda_P, A0, P0, 
                              J_diff, T, primes, hashes,
                              swap_neighborhood, moore_indices)
    return I, A, P


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def get_dH(s, dP, dA, A, P, lambda_A, lambda_P, A0, P0):
    """
    Calculate the change in energy.

    @param s: Index of the pixel in question.
    @param dP: Change in perimeter of that cell.
    @param dA: Change in area.
    @param A: Vector of areas, indexed with respect to cell indices prescribed in I.
    @param P: Vector of cell perimeters, indexed with respect to cell indices prescribed in I.
    @param lambda_A: The coefficient for the (A-A0) term in the energy functional. Cell-wise.
    @param lambda_P: The coefficient for the (P-P0) term in the energy functional. Cell wise.
    @param A0: Optimal area for each cell. Cell wise.
    @param P0: Optimal perimeter for each cell. Cell wise.

    @return: change in energy dH
    """
    A_old, P_old = A[s], P[s]
    A_new, P_new = A_old + dA, P_old + dP
    
    lambda_A_s, lambda_P_s = lambda_A[s], lambda_P[s]
    A0_s, P0_s = A0[s], P0[s]
    
    dA_new = A_new - A0_s
    dP_new = P_new - P0_s
    H_new = lambda_A_s * dA_new * dA_new + lambda_P_s * dP_new * dP_new
    
    dA_old = A_old - A0_s
    dP_old = P_old - P0_s
    H_old = lambda_A_s * dA_old * dA_old + lambda_P_s * dP_old * dP_old
    
    return H_new - H_old


@jit(nopython=True, cache=True, inline='always')
def pick_pixel(I, num_x, num_y, neighborhood):
    """
    Algorithm to choose pixels.
    This is the original Monte Carlo approach, that picks any
    pixel in the grid and leads to multiple rejections.

    1. Randomly choose a pixel (i,j) within boundaries.
    2. Define the cell index of the pixel (i,j) as s
    3. Pick one of the indices of the neighbouring pixels. s2.

    @param I: The sigma_field, (num_x x num_y) matrix of ints.
    @param num_x: Number of pixels in the x-dimension of I.
    @param num_y: Number of pixels in the y-dimension of I.

    @return: i: Chosen pixel x-component.
             j: Chosen pixel y-component.
             s: Index of the pixel in question.
            s2: Index of the neighbouring.
    """

    # Define valid range to avoid boundary pixels
    min_coord = 1
    max_x = num_x - 1
    max_y = num_y - 1

    # Generate random coordinates in valid range
    i = min_coord + int(np.random.random() * (max_x - min_coord))
    j = min_coord + int(np.random.random() * (max_y - min_coord))

    s = I[i, j]

    # Pick random neighbor from neighborhood
    neighbor_idx = int(np.random.random() * len(neighborhood))
    ni_x, ni_y = neighborhood[neighbor_idx]

    # Calculate neighbor coordinates with wrapping (periodic boundary)
    ni = (i + ni_x) % num_x
    nj = (j + ni_y) % num_y
    s2 = I[ni, nj]

    return i, j, s, s2


@jit(nopython=True, cache=True, fastmath = True, inline='always')
def get_mask_id(the_mask, primes, hashes):
    """
    The mask Na==s generically defines the neighbourhood. 
    Only certain masks are allowed in order to preserve
    local Moore contiguity and hence global Moore contiguity.

    This is achieved by indexing the hashes of the allowed masks.
    mask_id is the index of the mask Na==s 
    within the pre-defined list of acceptable masks.

    @param the_mask: A 3x3 boolean array defining the neighbourhood of a pixel. 
                     In the code this is Na==s or Na==s2.
    @param primes: The kernel used to hash the mask.
    @param hashes: The list of acceptable hashes.

    @return: This function returns the index of the mask from the list of acceptable masks.
             If the_mask is not in this list, -1 is returned.
    """
    
    hash_value = 0
    for i in range(3):
        for j in range(3):
            hash_value += the_mask[i, j]*primes[i, j]
         
    n_hashes = len(hashes)   
    k = 0

    while (k < n_hashes):
        if hash_value == hashes[k]:
            return(k)
        else:
            k += 1
    return -1


@jit(nopython=True, cache=True, inline='always')
def get_Na(I, i, j):
    """
    Given a point (i,j), subset the sigma field to find the 
    Moore neighbourhood. Na is a matrix of size (3,3)
    @param I: Sigma field (num_x x num_y) matrix of ints.
    @param i: Chosen pixel x-component.
    @param j: Chosen pixel y-component.
    @return: Na. The 3x3 subset I centred on (i,j).
    """
    Na = I[i - 1:i + 2, j - 1:j + 2]
    return Na


@jit(nopython=True, cache=True, fastmath=True, inline='always')
def get_dJ(J_diff, s, s2, Na, moore_indices):
    """
    Calculate the change in the interfacial energy, dJ.

    @param J_diff: Change in the interfacial energy when a pixel 
                   is replaced from cell index i to cell index j.
    @param s: Index of the pixel in question.
    @param s2: Index of the neighbouring pixel.
    @param Na: A 3x3 matrix, subsetting the sigma field centred on (i,j).

    @return: dJ -- change in J matrix
    """
    Js2s = J_diff[s2, s]

    # Select and sum the interfacial energy values corresponding to 
    # the cell indices in the neighborhood (excluding the center)
    dJ = 0.0
    for idx in moore_indices:
        row = idx // 3
        col = idx % 3
        neighbor_cell = Na[row, col]
        dJ += Js2s[neighbor_cell]

    return dJ
