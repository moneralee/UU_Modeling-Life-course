#!/usr/bin/env python3

"""
This module establishes the set of masks -- the 'zmasks' -- 
whose purpose is to enforce cell connectivity in all the copy attempts.
In essence, this prevents cellular fragmentation, 
which can be particularly pathological under high temperatures.

The idea is to check local Neumann neighbor connectivity within
a local Moore neighborhood of a pixel. There are only a
limited amount of possible cell configurations in a 3x3 grid that
respect local connectivity. The class Zmasks generates valid masks 
corresponding to these permissive configurations and
a hash table to efficiently access the masks.

Based on the following paper:
Durand, Marc, and Etienne Guesnet. 
"An efficient Cellular Potts Model algorithm that forbids cell fragmentation." 
Computer Physics Communications 208 (2016): 54-63.
"""

# Installed modules
import numpy as np

####################
# Class Definition #
####################

class Zmasks:
    """
    Defines the Zmasks class, wrapping the identities of 
    the zmasks, the hashes, and corresponding changes in perimeter
    and area.

    Each zmask has pre-defined changes in area and perimeter, 
    meaning these can be pre-calculated.

    The overall M-H algorithm proceeds by indexing the acceptable masks, 
    and using this pre-calculated change.

    Indexing is achieved by a hashing schema:
    - If Na = a 3x3 matrix of cell ids centred on a chosen pixel (i,j) 
      by subsetting the sigma_field matrix.
    - Local neighborhood is defined as a boolean 3x3 matrix. 
      This is either Na == s or Na == s2. s is the cell id of the
      pixel (i,j), whereas s2 is the cell id chosen to be the neighbour, 
      participating in a putative swap.
    - This boolean 3x3 matrix is the zmask.
    - zmasks are hashed by multiplying element-wise a (3x3) matrix, 
      which contains powers of 2. This is the matrix **primes**.
    - The hash is the sum of the element-wise multiplication between zmask and primes.
    """

    def __init__(self):
        """
        Initialisation of the Zmasks class.
        Follows the schema in Durand and Guesnet, 2016.
        """
        # List of rotations (which are contiguity invariant), defined in the functions below
        rots = [self.r0, self.r90, self.r180, self.r270]

        # Define z2_masks. 
        # z2_masks are the subset of acceptable 
        # z_masks that contain exactly 2 True values in the
        # Neumann neighborhood (ignoring the central pixel (1,1)). 
        self.z2_mask = np.zeros([3, 3], dtype=bool)
        self.z2_mask[0, 1], self.z2_mask[1, 0], self.z2_mask[0, 0] = True, True, True
        
        self.z2_mask2 = self.z2_mask.copy()
        self.z2_mask2[0, 2] = True

        self.z2_mask3 = self.z2_mask2.T

        self.z2_mask4 = self.z2_mask2.copy()
        self.z2_mask4[2, 0] = True

        # Apply rotations
        z2_base_masks = [self.z2_mask, self.z2_mask2, self.z2_mask3, self.z2_mask4]
        z2_rotated = []
        for mask in z2_base_masks:
            z2_rotated.extend([rot(mask) for rot in rots])
        self.z2_masks = np.array(z2_rotated)
        
        # Make middle pixel true
        self.z2_masks_ = self.z2_masks.copy()
        self.z2_masks_[:, 1, 1] = True

        self.z2_masks = np.concatenate([self.z2_masks, self.z2_masks_])

        # Define z3_masks. 
        # z3_masks are the subset of acceptable 
        # z_masks that contain exactly 3 True values in the
        # Neumann neighborhood (ignoring the central cell (1,1)). 
        self.z3_mask = np.ones([3, 3], dtype=bool)
        self.z3_mask[:, 2] = False
        self.z3_mask[1, 1] = False

        self.z3_mask2 = self.z3_mask.copy()
        self.z3_mask2[0, 2] = True

        self.z3_mask3 = self.z3_mask2.T
        
        # Apply rotations
        z3_base_masks = [self.z3_mask, self.z3_mask2, self.z3_mask3]
        z3_rotated = []
        for mask in z3_base_masks:
            z3_rotated.extend([rot(mask) for rot in rots])
        self.z3_masks = np.array(z3_rotated)

        # Make middle pixel true
        self.z3_masks_ = self.z3_masks.copy()
        self.z3_masks_[:, 1, 1] = True

        self.z3_masks = np.concatenate([self.z3_masks, self.z3_masks_])

        # Define z1_masks. 
        # z1_masks are the subset of acceptable z_masks 
        # that contain exactly 1 True value in the
        # Neumann neighborhood (ignoring the central cell (1,1)). 
        self.z1_mask = np.zeros([3, 3], dtype=bool)
        self.z1_mask[0, 1] = True

        self.z1_mask2 = self.z1_mask.copy()
        self.z1_mask2[0, 2] = True

        self.z1_mask3 = self.z1_mask2.T

        self.z1_mask4 = self.z1_mask2.copy()
        self.z1_mask4[0, 0] = True

        # Apply rotations
        z1_base_masks = [self.z1_mask, self.z1_mask2, self.z1_mask3, self.z1_mask4]
        z1_rotated = []
        for mask in z1_base_masks:
            z1_rotated.extend([rot(mask) for rot in rots])
        self.z1_masks = np.array(z1_rotated)
        
        # Make middle pixel true
        self.z1_masks_ = self.z1_masks.copy()
        self.z1_masks_[:, 1, 1] = True
        self.z1_masks = np.concatenate([self.z1_masks, self.z1_masks_])

        # z_masks is a (n_zmasks x 3 x 3) boolean array, 
        # generated by combining z1_masks, z2_masks, and z3_masks.
        self.z_masks = np.concatenate((self.z1_masks, self.z2_masks, self.z3_masks))

        # Calculate the local perimeter (changes) under each of the masks. 
        # Positions of True correspond to the cell in question.
        i_, j_ = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2), indexing="ij")
        self.Moore = np.column_stack([i_.ravel(), j_.ravel()])

        # Pre-compute indices excluding center
        center_mask = np.ones(9, dtype=bool)
        center_mask[4] = False  # Center index in flattened 3x3
        self.perim_neighbour = self.Moore[center_mask]

        # Calculate the local perimeter changes under each of the masks
        self.get_dP_masks()

        # Establish the hashing for the acceptable z_masks. 
        self.primes = 2 ** np.arange(9).reshape(3, 3)
        self.hashes = np.sum(self.z_masks.astype(np.int32) * self.primes, axis=(1, 2))


    def r0(self, x):
        """
        A 0-degree rotation of the matrix x.
        @param x: Input matrix.
        @return: 0-degree rotation
        """
        return x


    def r90(self, x):
        """
        A 90-degree rotation of the matrix x.
        @param x: Input matrix.
        @return: 90-degree rotation
        """
        return np.flip(x.T, axis=1)
    

    def r180(self, x):
        """
        A 180-degree rotation of the matrix x.
        @param x: Input matrix.
        @return: 180-degree rotation
        """
        return np.flip(np.flip(x, axis=0), axis=1)
    

    def r270(self, x):
        """
        A 270-degree rotation of the matrix x.
        @param x: Input matrix.
        @return: 270-degree rotation
        """
        return np.flip(x.T, axis=0)


    def get_dP_masks(self):
        """
        Calculate the change in perimeter for the **True** cell in each of the zmasks.

        This pre-calculation of the change in perimeter massively speeds up the CPM calculations.
        """

        def getPA(masks):
            n = masks.shape[0] // 2
            # Vectorized perimeter calculation for all masks at once
            P_first_half = np.array([self.get_perimeter_and_area_not_periodic(masks[i], True)[0] for i in range(n)])
            P_second_half = np.array([self.get_perimeter_and_area_not_periodic(masks[n + i], True)[0] for i in range(n)])
            P_diff = P_second_half - P_first_half
            return np.concatenate([P_diff, -P_diff])

        self.z1_masks_dP = getPA(self.z1_masks)
        self.z2_masks_dP = getPA(self.z2_masks)
        self.z3_masks_dP = getPA(self.z3_masks)
        self.dP_z = np.concatenate((self.z1_masks_dP, self.z2_masks_dP, self.z3_masks_dP))


    def get_perimeter_and_area_not_periodic(self, I, s):
        """
        Calculate the area and perimeter of a cell, given a matrix of cell_ids.

        @param I: Sigma field. Matrix of cell ids.
        @param s: Cell id for which area and perimeter is to be calculated.
        @return: Perimeter of the cell. Area of the cell.
        """
        M = I == s
        
        P = 0
        for di, dj in self.perim_neighbour:
            # Use modulo for boundary wrapping
            shifted_i = (np.arange(3) + di) % 3
            shifted_j = (np.arange(3) + dj) % 3
            shifted = M[np.ix_(shifted_i, shifted_j)]

            # Only count interior (1:-1, 1:-1) but for 3x3 that's just (1,1)
            P += int(M[1, 1] != shifted[1, 1])

        A = np.sum(M)
        return P, A