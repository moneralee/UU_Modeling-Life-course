import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create a matrix with random values within a rectangle
minx = 3
miny = 2
maxx = 6
maxy = 7

input = np.zeros((10, 10))
input[minx:maxx, miny:maxy] = np.random.randn(maxx - minx, maxy - miny) + 1

# Create bigL matrix
bigL = np.zeros((input.shape[0] + 2, input.shape[1] + 2))

# Copy original matrix and pad 1 pixel-wide boundary with edge values
bigL[minx:maxx+2, miny:maxy+2] = np.pad(input[minx:maxx, miny:maxy], 1, mode='edge')

# Calculate the Laplacian
minX, maxX, minY, maxY = minx+1, maxx+1, miny+1, maxy+1
laplacian = np.zeros_like(input)

shifted_up    = bigL[minX-1:maxX-1, minY:maxY]
shifted_down  = bigL[minX+1:maxX+1, minY:maxY]
shifted_right = bigL[minX:maxX, minY+1:maxY+1]
shifted_left  = bigL[minX:maxX, minY-1:maxY-1]

laplacian[minx:maxx, miny:maxy] = (
    shifted_up +
    shifted_down +
    shifted_right +
    shifted_left -
    4 * bigL[minX:maxX, minY:maxY]
)


### Plotting ###
fig, axes = plt.subplots(2, 4, figsize=(12, 10))

# Collect all matrices for global min/max
matrices = [input, bigL, shifted_up, shifted_down, shifted_right, shifted_left]
global_min = min(m.min() for m in matrices)
global_max = max(m.max() for m in matrices)

# Upper row of plots
im0 = axes[0, 0].imshow(input, cmap='viridis', vmin=global_min, vmax=global_max)
axes[0, 0].set_title("input")

im1 = axes[0, 1].imshow(bigL, cmap='viridis', vmin=global_min, vmax=global_max)
axes[0, 1].set_title("bigL")

im2 = axes[0, 2].imshow(laplacian, cmap='plasma')
axes[0, 2].set_title("laplacian")

# The final plot in the upper row shows the relative position 
# of input and shifted matrices. To do this, we create binary matrices first.
ax = axes[0, 3]

shifted_down_loc  = np.zeros_like(input)
shifted_up_loc    = np.zeros_like(input)
shifted_left_loc  = np.zeros_like(input)
shifted_right_loc = np.zeros_like(input)

shifted_down_loc[minx+1:maxx+1, miny:maxy]  = 1
shifted_up_loc[minx-1:maxx-1, miny:maxy]    = 1
shifted_left_loc[minx:maxx, miny-1:maxy-1]  = 1
shifted_right_loc[minx:maxx, miny+1:maxy+1] = 1

a = 0.2
ax.imshow(np.where(input!=0, 1, 0), cmap='binary', alpha=a, vmin=0, vmax=1)
ax.imshow(shifted_down_loc , cmap='Reds', alpha=a, vmin=0, vmax=1)
ax.imshow(shifted_up_loc   , cmap='Blues', alpha=a, vmin=0, vmax=1)
ax.imshow(shifted_left_loc , cmap='Greens', alpha=a, vmin=0, vmax=1)
ax.imshow(shifted_right_loc, cmap='Purples', alpha=a, vmin=0, vmax=1)

# Add legend
legend_elements = [
    Patch(facecolor='red', alpha=a, label='Shifted Down'),
    Patch(facecolor='blue'  , alpha=a, label='Shifted Up'),
    Patch(facecolor='green' , alpha=a, label='Shifted Left'),
    Patch(facecolor='purple', alpha=a, label='Shifted Right'),
    Patch(facecolor='gray'  , alpha=a, label='Original Input')
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.75))
ax.set_title("overlay of relative positions\nof input and shifted matrices")

# Lower row of plots
im3 = axes[1, 0].imshow(shifted_up, cmap='viridis', vmin=global_min, vmax=global_max)
axes[1, 0].set_title("bigL shifted up")

im4 = axes[1, 1].imshow(shifted_down, cmap='viridis', vmin=global_min, vmax=global_max)
axes[1, 1].set_title("bigL shifted down")

im5 = axes[1, 2].imshow(shifted_right, cmap='viridis', vmin=global_min, vmax=global_max)
axes[1, 2].set_title("bigL shifted right")

im6 = axes[1, 3].imshow(shifted_left, cmap='viridis', vmin=global_min, vmax=global_max)
axes[1, 3].set_title("bigL shifted left")

# Add colorbars for the first three plots
for i in range(3):
    divider = make_axes_locatable(axes[0, i])
    cax = divider.append_axes("bottom", size="10%", pad=0.1)
    fig.colorbar(axes[0, i].images[0], cax=cax, orientation='horizontal')

plt.tight_layout()
plt.show()