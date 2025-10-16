import numpy as np
import matplotlib.pyplot as plt

def sum_mask_shift(mask):
    """
    
    """
    
    mask_shift_sum = np.zeros_like(mask)

    # add shifted masks
    mask_shift_sum[:,1:]   = mask[:,:-1]
    mask_shift_sum[:,:-1] += mask[:,1:]
    mask_shift_sum[1:,:]  += mask[:-1,:]
    mask_shift_sum[:-1,:] += mask[1:,:]

    return(mask_shift_sum)


# Create a sample tissue mask for demonstration
tissue_mask = np.zeros((50, 50))
tissue_mask[12:33, 7:45] = 1  # A rectangular region as a sample tissue

mask_shift_sum = sum_mask_shift(tissue_mask)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(10, 8))

# Original tissue mask
axes[0].imshow(tissue_mask, cmap='gray')
axes[0].set_title('Original Tissue Mask')

# Binary erosion result
axes[1].imshow(mask_shift_sum, cmap='viridis')
axes[1].set_title('mask_shift_sum')


plt.tight_layout()
plt.show()