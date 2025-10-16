
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage


def create_paddle_mask(full_size_x, full_size_y, 
                       Ly, 
                       body_wall_x = 5,
                       arm_y = 25):
    """
    Create a tissue mask that is a combination of:
    - a tall rectangle for the "body"
    - a small rectangle for the "arm"
    - a circle for the "hand" that overlaps the "arm"

    Parameters:
        full_size_x, full_size_y (int): Size of the entire simulated domain.
        L_y (int): Length of arm and radius of hand
        body_wall_x (int): Length of body
        arm_y (int): Height of the arm

    Returns:
        np.ndarray: Binary mask (same shape as simulated domain), 
                    1 for points inside domain, 0 otherwise.
        int: x-coordinate of the hand center
        int: y-coordinate of the hand center
    """

    # Initialize matrix
    mask_matrix = np.zeros((full_size_x, full_size_y))

    # Define a rectangle of length body_wall_x and height full_size_y-10
    # Do not fill the entire height, otherwise the growth won't work
    mask_matrix[ 0:body_wall_x, 5:-5] = 1

    # Define a rectangle for the arm
    arm_length = Ly
    half_y = full_size_y//2
    mask_matrix[ body_wall_x:arm_length, half_y-arm_y//2:half_y+arm_y//2 ] = 1

    # Define a circle for the hand
    hand_center_x = int(0.9*arm_length + Ly) # slight overlap
    hand_center_y = half_y

    x,y = np.ogrid[ -hand_center_x:full_size_x-hand_center_x, 
                    -hand_center_y:full_size_y-hand_center_y ]

    in_circle = ( x**2 / Ly**2 ) + ( y**2 / Ly**2 ) <= 1

    mask_matrix[in_circle] = 1

    # Erode then apply a blur filter to smooth the corners
    mask_matrix = ndimage.binary_erosion(mask_matrix, iterations = 2)
    mask_matrix = ndimage.gaussian_filter(mask_matrix.astype(float), sigma=1)
    mask_matrix = np.where(mask_matrix > 0, 1, 0)
    mask_matrix = ndimage.binary_erosion(mask_matrix, iterations = 1).astype(float)

    return(mask_matrix, hand_center_x, hand_center_y)
        

def create_ellipse_mask(full_size_x, full_size_y, 
                        E_x, E_y, 
                        E_ax_x, E_ax_y):
    """
    Create an elliptical domain mask.

    Parameters:
        full_size_x, full_size_y (int): Size of the entire simulated domain.
        E_x, E_y (int): Center coordinates of the ellipse.
        E_ax_x, E_ax_y (int): Semi-axes lengths of the ellipse

    Returns:
        np.ndarray: Binary mask (same shape as simulated domain), 
                    1 for points inside ellipse, 0 otherwise.

    Parametric equation for ellipse: (x-xo)^2/a^2 + (y-yo)^2/b^2 = 1
    """

    # Create grid coordinates around ellipse area
    x, y = np.ogrid[ -E_x:full_size_x-E_x , -E_y:full_size_y-E_y ]

    # Use the parametric equation of an ellipse to define mask
    in_ellipse = (x**2 / E_ax_x**2) + (y**2 / E_ax_y**2) <= 1

    # Extend to entire simulation domain with 0s outside ellipse
    mask_matrix = np.zeros((full_size_x, full_size_y))
    mask_matrix[in_ellipse] = 1

    return mask_matrix


# Parameters for tissue mask
full_size_x, full_size_y = 100, 100
Ly = 20
body_wall_x = 5
arm_y = 25

# Parameters for ellipse mask
E_x, E_y = 50, 50
E_ax_x, E_ax_y = 30, 20

# Generate masks
tissue_mask, hand_center_x, hand_center_y = create_paddle_mask(full_size_x, full_size_y, Ly, body_wall_x, arm_y)
ellipse_mask = create_ellipse_mask(full_size_x, full_size_y, E_x, E_y, E_ax_x, E_ax_y)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot tissue mask
axes[0].imshow(tissue_mask.T, cmap='viridis')
axes[0].scatter(hand_center_x, hand_center_y, color='red', s=50, label='Hand Center')
axes[0].set_title('Tissue Mask')
axes[0].legend()

# Plot ellipse mask
axes[1].imshow(ellipse_mask.T, cmap='viridis')
axes[1].scatter(E_x, E_y, color='red', s=50, label='Ellipse Center')
axes[1].set_title('Ellipse Mask')
axes[1].legend()

plt.tight_layout()
plt.show()