import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def sum_mask_shift(mask):
    """
    Shifts the mask up/down/left/right and sums it.
    The result is a matrix where for each pixel,
    the value in that pixel is the number of
    valid flux neighbours for the diffusion calculation.
    """
    
    mask_shift_sum = np.zeros_like(mask)

    # Add shifted masks
    mask_shift_sum[  :  , 1:  ]  = mask[  :  ,  :-1]
    mask_shift_sum[  :  ,  :-1] += mask[  :  , 1:  ]
    mask_shift_sum[ 1:  ,  :  ] += mask[  :-1,  :  ]
    mask_shift_sum[  :-1,  :  ] += mask[ 1:  ,  :  ]

    return(mask_shift_sum)


# Create a sample tissue mask for demonstration
tissue_mask = np.zeros((25, 25))
tissue_mask[6:16, 7:21] = 1  # A rectangular region as a sample tissue

# Parameters
iterations = 2

# Apply binary erosion
internal_mask = ndimage.binary_erosion(tissue_mask, iterations=iterations)

# Extract boundary points
internal_mask = ndimage.binary_erosion(tissue_mask).astype(tissue_mask.dtype)
boundary_mask = tissue_mask - internal_mask

# Get shifted mask sum
mask_shift_sum = sum_mask_shift(tissue_mask)

# Get boundary coordinates
boundary_coords = np.argwhere(boundary_mask != 0)

# Visualization
fig, axes = plt.subplots(2, 3, figsize=(10, 8))

# Original tissue mask
axes[0,0].imshow(tissue_mask, cmap='gray')
axes[0,0].set_title('Original Tissue Mask')

# Binary erosion result
axes[0,1].imshow(ndimage.binary_erosion(tissue_mask, iterations=2), cmap='gray')
axes[0,1].set_title('Binary Erosion of tissue mask')

# Binary dilation result
axes[0,2].imshow(ndimage.binary_dilation(tissue_mask, iterations=2), cmap='gray')
axes[0,2].set_title('Binary Dilation of tissue mask')

# Boundary mask
axes[1,0].imshow(boundary_mask, cmap='gray')
axes[1,0].set_title('Boundary Mask')

# Boundary coordinates on top of shifted mask sum
bound_x, bound_y = boundary_coords[:, 1], boundary_coords[:, 0]

axes[1,1].imshow(mask_shift_sum, cmap='viridis')
axes[1,1].scatter(bound_x, bound_y, c='blue', s=2, label="Boundary Point")

axes[1,1].set_title('Shifted mask sum = valid flux neighbors')
axes[1,1].legend(loc='lower right')

# Gaussian filter result
axes[1,2].imshow(ndimage.gaussian_filter(boundary_mask.astype(float), sigma=1), cmap='gray')
axes[1,2].set_title('Gaussian Filter on boundary mask')

plt.tight_layout()
plt.show()