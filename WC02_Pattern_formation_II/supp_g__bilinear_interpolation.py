import numpy as np
import matplotlib.pyplot as plt

def growth_bilinear_interpolation(input_matrix, old_bounds, new_bounds):
    """
    Use bilinear interpolation to grow the input domain to the new size.
    """ 

    minxO, maxxO, minyO, maxyO = old_bounds
    minxN, maxxN, minyN, maxyN = new_bounds

    xrangeO = maxxO - minxO
    yrangeO = maxyO - minyO

    xrangeN = maxxN - minxN
    yrangeN = maxyN - minyN

    # No growth, no need to calculate
    if ( xrangeO - xrangeN == 0 ) and ( yrangeO - yrangeN == 0 ):
        return(input_matrix)
    
    # Extract tissue values
    inner_matrix = input_matrix[minxO:maxxO, minyO:maxyO]

    # Map new matrix indices to old matrix indices
    new_x = np.linspace(0, 1, xrangeN)
    new_y = np.linspace(0, 1, yrangeN)
    x_idx = new_x * (inner_matrix.shape[0] - 1)
    y_idx = new_y * (inner_matrix.shape[1] - 1)
    x_idx_grid, y_idx_grid = np.meshgrid(x_idx, y_idx, indexing='ij')

    # Vectorized bilinear interpolation
    # Determine surrounding points in old matrix indexing
    # Lower indices
    x0 = np.floor(x_idx_grid).astype(int) 
    y0 = np.floor(y_idx_grid).astype(int)

    # Higher indices, clipped to stay within bounds
    x1 = np.clip(x0 + 1, 0, inner_matrix.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, inner_matrix.shape[1] - 1)

    # Intermediate points to interpolate on
    xd = x_idx_grid - x0
    yd = y_idx_grid - y0

    # Ia, Ib, Ic, Id are the values at the four corners of the old grid 
    # surrounding each new point.
    Ia = inner_matrix[x0, y0]
    Ib = inner_matrix[x1, y0]
    Ic = inner_matrix[x0, y1]
    Id = inner_matrix[x1, y1]

    # The new values are a weighted average of the four corners, 
    # based on how close each new point is to each corner.
    out_matrix = (Ia * (1 - xd) * (1 - yd) +
                  Ib * xd * (1 - yd) +
                  Ic * (1 - xd) * yd +
                  Id * xd * yd)

    output_matrix = np.zeros_like(input_matrix)
    output_matrix[minxN:maxxN, minyN:maxyN] = out_matrix

    return output_matrix


# Create a matrix with random values within a rectangle
minx = 3
miny = 2
maxx = 6
maxy = 7

input = np.zeros((10, 10))
input[minx:maxx, miny:maxy] = np.random.randn(maxx - minx, maxy - miny) + 1

old_bounds = (minx, maxx, miny, maxy)
new_bounds = (minx-2, maxx+1, miny-1, maxy+2)

tissue_after_growth = growth_bilinear_interpolation(input, old_bounds, new_bounds)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot the tissue before growth
axes[0].imshow(input, cmap='viridis')
axes[0].set_title('Tissue before growth')

# Plot the tissue after growth
axes[1].imshow(tissue_after_growth, cmap='viridis')
axes[1].set_title('Tissue after growth')


plt.tight_layout()
plt.show()