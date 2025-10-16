import numpy as np
import matplotlib.pyplot as plt

def ellipse_slice(array, Ex, Ey, Lx, Ly, theta, max_radius=1, min_radius = 0):
    """
    Generate a binary mask for an elliptical slice in a 2D array.

    Parameters:
        array (np.ndarray): Input 2D array (shape used for mask).
        Ex, Ey (float): Center coordinates of the ellipse.
        Lx, Ly (float): Semi-axes lengths of the ellipse.
        theta (float): Angular width of the slice (radians, centered on +x axis).
        max_radius (float, optional): Maximum radius from center to include (default is 1).
        min_radius (float, optional): Minimum radius from center to include (default is 0).

    Returns:
        np.ndarray: Binary mask (same shape as array), 1 for points inside slice, 0 otherwise.
    """
    rows, cols = array.shape

    # Create coordinate grid
    y, x = np.ogrid[:rows, :cols]

    # Translate coordinates to ellipse center
    dx = x - Ex
    dy = y - Ey

    # Calculate squared distances and angles
    angles = np.arctan2(dy, dx)

    # Ellipse equation
    in_ellipse_max = (dx**2 / Lx**2) + (dy**2 / Ly**2) <= max_radius**2
    in_ellipse_min = (dx**2 / Lx**2) + (dy**2 / Ly**2) >= min_radius**2

    # Slice condition
    in_slice = (angles >= -theta/2) & (angles <= theta/2)

    # Combine all conditions
    mask = in_ellipse_max & in_ellipse_min & in_slice

    return mask.astype(int)


# Generate a matrix of random values
random_matrix = np.random.random((40, 40))

# Create a linear gradient from top-left to bottom-right to make pattern more visible
x = np.linspace(0, 1, random_matrix.shape[1])
y = np.linspace(0, 1, random_matrix.shape[0])
X, Y = np.meshgrid(x, y)
gradient = X + Y  # Simple linear gradient

array = random_matrix*gradient


# Define ellipse
Ex, Ey = 7, 15     # Center of the ellipse
Lx, Ly = 5, 3      # Semi-axes lengths
theta = np.pi / 4  # Angle of the slice
max_radius = 6     # Optional maximum radius
min_radius = 3     # Optional minimum radius

# Call the function
binary_mask = ellipse_slice(array, Ex, Ey, Lx, Ly, theta, max_radius, min_radius)


# Plotting
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# Plot all data
axes[0].imshow(array, cmap='viridis', vmin = 0, vmax = 1)
axes[0].set_title('All data')
axes[0].set_aspect('equal')

# Plot mask
axes[1].imshow(binary_mask, cmap='Reds', alpha=0.7)
axes[1].set_title('Ellipse Slice Mask')
axes[1].set_aspect('equal')

# Masked data
axes[2].imshow(binary_mask*array, cmap='viridis', vmin = 0, vmax = 1)
axes[2].set_title('Masked data')
axes[2].set_aspect('equal')

plt.show()