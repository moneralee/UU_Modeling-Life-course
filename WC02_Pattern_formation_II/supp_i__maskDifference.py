import numpy as np
import matplotlib.pyplot as plt

# Create a binary mask with a non-centered rectangle
output_mask = np.zeros((12, 12))
output_mask[2:9, 4:7] = True  # Rectangle from (5,8) to (11,14)

# Create a smaller mask
input_smaller = np.zeros((12, 12))
input_smaller[2:8, 4:7] = 1

# Create a bigger mask
input_bigger = np.zeros((12, 12))
input_bigger[2:10, 4:7] = 1

print("Difference output mask - smaller input mask:", np.sum(output_mask - input_smaller))
print("Difference output mask - larger input mask:",np.sum(output_mask - input_bigger))

# Plot the masks side by side
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

axes[0, 0].imshow(output_mask.T, cmap='Grays')
axes[0, 0].set_title("Original Mask")

axes[0, 1].imshow(input_smaller.T, cmap='Reds')
axes[0, 1].set_title("Smaller Mask")

# Subtract: Mask 1 - Smaller Mask
axes[0, 2].imshow((output_mask-input_smaller).T, cmap='Reds')
axes[0, 2].set_title("output mask - smaller input mask")

axes[1, 0].imshow(output_mask.T, cmap='Grays')
axes[1, 0].set_title("Original Mask")

axes[1, 1].imshow(input_bigger.T, cmap='Greens')
axes[1, 1].set_title("Bigger Mask")

# Subtract: Mask 1 - Bigger Mask
axes[1, 2].imshow((output_mask-input_bigger).T, cmap='Greens')
axes[1, 2].set_title("output mask - larger input mask")
plt.show()