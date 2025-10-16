"""
Implements the BSW model from Raspopovic et al. (2014) in a growing Ellipse shape.

References:
- Raspopovic, Jelena, et al. "Digit patterning is controlled by a 
  Bmp-Sox9-Wnt Turing network modulated by morphogen gradients." 
  Science 345.6196 (2014): 566-570.
"""

# System modules
import sys

# Installed modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import ndimage

##############
# PARAMETERS #
##############

# Space discretization
nx = 60    # Grid size in x-direction
ny = 60    # Grid size in y-direction
dx = 0.05  # Distance between grid points

# Time discretization
dt = 0.1
totaltime = 2000
time_steps = int(totaltime / dt + 1)

# Model parameters
# See Table ST7 in supplementary information of Raspopovic et al. (2014)
# Reaction rate parameters
k2 = 1    
k3 = 1    
k4 = 1.95  # NOTE new value! old: 1.59
k5 = 0.1  
k7 = 0.55  # NOTE new value! old: 1.27
k9 = 0.1  
delta = 0

# "BSW model can form a relatively robust Turing pattern with a difference in
# diffusivity of two and a half (d = 2.5) between Bmp and Wnt." - Raspopovic et al. (2014)
Dwnt = 0.001
Dbmp = 2.5*Dwnt 

# Parameters for FGF and Hoxd13 modulation of the BSW model
# NOTE: If set to 0, no pattern forms!
k_HF_bmp = 0.36
k_HF_wnt = 0.72


# Horizontal and vertical growth rates
vi = 0.01 # NOTE: does nothing for geometry "paddle"
vj = 0.008

# Initial condition parameters for domain
# Horizontal: Start on left side of the nx*ny box
Lx0 = 20

max_domain_horizontal = int(0.8*nx) # Stop growing once this reached

# Vertical: Start in middle of the nx*ny box
Ly0 = 10

max_domain_vertical = int(0.7*ny) # Stop growing once this reached

# Tissue growth type
# 'add' - tissue grows by incorporating new material with steady state value
# 'stretch' - tissue grows by stretching existing domain values
growth_type = 'stretch'

# Tissue geometry type
# 'ellipse' - a simple ellipse
# 'paddle'  - two rectangles for body and arm and a circle for the hand
geometry = 'paddle' 
ellipse_center_x = 10    # only for ellipse geometry
ellipse_center_y = ny//2 # only for ellipse geometry


# Parameters for FGF and Hoxd13 modulation of the BSW model
# In the original paper, the authors use experimental data 
# to map expression patterns to the simulated tissue geometry.
# Here, we use a rough approximation to reproduce the patterns.

# Reaction-diffusion parameters for FGF
Dfgf = 0.001            # diffusion coefficient
fgf_decay_rate = 0.0003 # degradation parameter

# Opening angle of fgf source at the start and end
fgf_angle_start_1 = -0.3*np.pi
fgf_angle_start_2 = -0.5*np.pi
fgf_angle_end_1   =  0.4*np.pi
fgf_angle_end_2   = -0.6*np.pi    
fgf_angle_rate    = 0.0002      # Exponential rate of change of angle

# Opening angle of Hoxd13 expression domain at the start and end
hox_angle_start_1 = -0.1*np.pi
hox_angle_start_2 = -0.3*np.pi
hox_angle_end_1   =  0.6*np.pi
hox_angle_end_2   = -0.6*np.pi
hox_angle_rate    = 0.001       # Exponential rate of change of angle

hox_minR   = 0.3    # Minimum radius for Hoxd13 permissive zone at the end
hox_maxR   = 0.85   # Maximum radius for Hoxd13 permissive zone (also minimum at the start)
hox_R_rate = 0.001  # Rate of change of the Hoxd13 minimum radius over time


#####################
# CLASS DEFINITIONS #
#####################

class DomainMask:
    """
    Base class for domain masks.
    Implements common functions needed by both types of masks.
    """

    def __init__(self, geometry, Lx, Ly, nx, ny):
        """
        Initialize a new mask object with default parameter values.
        """

        # Check if the user selected a valid geometry
        self.geometry = geometry

        if self.geometry not in ['ellipse','paddle']:
            print("Undefined geometry given, choose 'ellipse' or 'paddle':", self.geometry)
            sys.exit()

        # Set initial domain size
        self.Lx = Lx
        self.Ly = Ly

        # Set grid size
        self.nx = nx
        self.ny = ny


    def update(self):
        """
        Update mask to new size upon tissue growth.
        - update internal and boundary mask
        - update valid flux neighbours
        """

        # The internal mask excludes a 1-pixel wide boundary
        self.internal_mask = ndimage.binary_erosion(self.mask, iterations = 1)

        # The boundary mask is the 1-pixel wide boundary
        self.boundary_mask = self.mask - self.internal_mask
        
        # Update valid flux neighbours with Class method sum_mask_shift()
        self.sum_mask_shift()


    def sum_mask_shift(self):
        """
        Shifts the mask up/down/left/right and sums it.
        The result is a matrix where for each pixel,
        the value in that pixel is the number of
        valid flux neighbours for the diffusion calculation.
        """
        
        self.mask_shift_sum = np.zeros_like(self.mask)

        # Add shifted masks
        self.mask_shift_sum[  :  , 1:  ]  = self.mask[  :  ,  :-1]
        self.mask_shift_sum[  :  ,  :-1] += self.mask[  :  , 1:  ]
        self.mask_shift_sum[ 1:  ,  :  ] += self.mask[  :-1,  :  ]
        self.mask_shift_sum[  :-1,  :  ] += self.mask[ 1:  ,  :  ]


    def copy(self):
        """
        Create a new mask that is a copy of this mask.
        This is used to store the tissue mask before growth.

        The method should be called from one of the subclasses
        that inherit from DomainMask.
        """

        new_mask = DomainMask(self.geometry, self.Lx, self.Ly, self.nx, self.ny)

        new_mask.mask   = self.mask.copy()
        new_mask.bounds = self.bounds.copy()

        new_mask.Lx = self.Lx
        new_mask.Ly = self.Ly

        new_mask.internal_mask = self.internal_mask.copy()
        new_mask.boundary_mask = self.boundary_mask.copy()

        # The following variables are initialized by subclasses
        new_mask.Cx = self.Cx
        new_mask.Cy = self.Cy

        if self.geometry == 'paddle':
            new_mask.body_wall_x = self.body_wall_x
            new_mask.arm_y = self.arm_y
        
        return new_mask
    

class EllipseMask(DomainMask):
    """
    Subclass for elliptical domain mask that inherits from DomainMask.
    """

    def __init__(self, geometry, Lx0, Ly0, nx, ny, ellipse_center_x, ellipse_center_y):

        # Call the parent class's __init__ method using the "super" keyword
        super().__init__(geometry, Lx0, Ly0, nx, ny)

        # Set center coordinates
        self.Cx = ellipse_center_x
        self.Cy = ellipse_center_y

        self.mask = create_ellipse_mask(self.nx, self.ny, 
                                        self.Cx, self.Cy, 
                                        Lx0, Ly0)
        
        # Define boundaries - hardcoded to improve computation speed 
        # These will be used to reduce the area of calculations for diffusion
        self.bounds = [max(0, self.Cx - Lx0), 
                       self.Cy + Lx0 + 1, 
                       self.Cx - Ly0, 
                       self.Cy + Ly0 + 1]
        
        # Create the boundary mask - a 1-pixel wide outline of the geometry
        self.internal_mask = ndimage.binary_erosion(self.mask, iterations = 1)
        self.boundary_mask = self.mask - self.internal_mask

        # Determine valid flux neighbours (saved in self.mask_shift_sum)
        self.sum_mask_shift()
    

    def update(self, domain_size_horizontal, domain_size_vertical, max_domain_horizontal, max_domain_vertical):
        """
        Update mask to new size upon tissue growth.
        """

        # Only grow horizontally if not yet reached maximum horizontal size
        if self.bounds[1] < max_domain_horizontal:
            self.Lx = domain_size_horizontal

        # Only grow vertically if not yet reached maximum vertical size
        if (self.bounds[3] - self.bounds[2]) < max_domain_vertical:
            self.Ly = domain_size_vertical

        self.mask = create_ellipse_mask(self.nx, self.ny, 
                                        self.Cx, self.Cy, 
                                        self.Lx, self.Ly)
        
        # Update bounds - hardcoded to improve computation speed
        self.bounds = [max(0,self.Cx - self.Lx), 
                       self.Cx + self.Lx + 1,
                       self.Cy - self.Ly, 
                       self.Cy + self.Ly + 1]
        
        # Run the parent class' update method
        super().update()
                

class PaddleMask(DomainMask):
    """
    Subclass for paddle-shaped domain mask that inherits from DomainMask.
    """

    def __init__(self, geometry, Lx0, Ly0, nx, ny, body_wall_x=5,arm_y=25):
        
        # Call the parent class's __init__ method using the "super" keyword
        super().__init__(geometry,Lx0,Ly0,nx,ny)

        # Create tissue mask
        self.body_wall_x = body_wall_x
        self.arm_y = arm_y
        self.mask, self.Cx, self.Cy = create_paddle_mask(nx, ny, Ly0, self.body_wall_x, self.arm_y)

        # Define bounds - hardcoded to improve computation speed
        # These will be used to reduce the area of calculations for diffusion

        # The maximum x coordinate consists of:
        # center position of the hand = int(0.9*self.Ly + self.Ly)
        # + one hand radius = self.Ly
        # +2 pixels to account for erode x2 + gauss x1   # HACK
        self.bounds = [0, 
                       int(0.9*self.Ly + self.Ly) + self.Ly + 2, 
                       0, 
                       ny-1]

        self.internal_mask = ndimage.binary_erosion(self.mask, iterations = 1)
        self.boundary_mask = self.mask - self.internal_mask
        self.sum_mask_shift()
    
    def update(self, domain_size_horizontal, domain_size_vertical, max_domain_horizontal, max_domain_vertical):
        """
        Update mask to new size upon tissue growth.
        """

        # The paddle mask is defined to grow only according to the horizontal growth
        # The radius of the circle representing the "hand" grows, thus horizontal = vertical growth
        if self.bounds[1] < max_domain_horizontal:
            Ly = domain_size_vertical

            if Ly > self.Ly:
                self.Ly = Ly
                self.mask, self.Cx, self.Cy = create_paddle_mask(self.nx, self.ny, self.Ly)

                # Update bounds - hardcoded to improve computation speed

                # The maximum x coordinate consists of:
                # center position of the hand = int(0.9*self.Ly + self.Ly)
                # + one hand radius = self.Ly
                # +2 pixels to account for erode x2 + gauss x1   # HACK
                self.bounds = [0, 
                               int(0.9*self.Ly + self.Ly) + self.Ly + 2, 
                               0, 
                               self.ny-1]
                
                # Run the parent class' update method
                super().update()


#############
# FUNCTIONS #
#############

def get_domain_size_horizontal(n):
    """
    Calculate the new horizontal domain size.
    Parameters:
        n (int) - timestep
    
    Global variables:
        nx (int)   - horizontal grid size
        Lx0 (int)  - initial horizontal domain size
        dt (float) - time step
        vi (float) - horizontal growth velocity
    """
    return min( round(Lx0 + dt * vi * n), nx)

def get_domain_size_vertical(n):
    """
    Calculate the new vertical domain size.
    Parameters:
        n (int) - timestep
    
    Global variables:
        ny (int)   - vertical grid size
        Ly0 (int)  - initial vertical domain size
        dt (float) - time step
        vj (float) - vertical growth velocity
    """
    return min( round(Ly0 + dt * vj * n), ny )


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

    # Define a rectangle of length body_wall_x and height equal to the grid
    mask_matrix[ 0:body_wall_x, :] = 1

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


def smooth_function(t, start, end, rate):
    """
    Function to create smooth rate of change of some variable
    from a start value to an end value with a given exponential rate.
    """
    return end + (start - end) * np.exp(-rate * t )


def compute_FGF_source_on_mask(t, tissue_mask):
    """
    Computes the location of the FGF source on a tissue mask.
    
    Parameters:
        tissue_mask (numpy.ndarray): A domain representing the tissue location.
        t (int) : Current time step

    Global variables:

        fgf_decay_rate_start  : Decay rate for FGF gradient at the start
        fgf_decay_rate_end    : Decay rate for FGF gradient at the end
        fgf_decay_rate_change : Exponential rate of change of decay rate
        
        fgf_angle_start : Angle of the FGF permissive zone when t=0
        fgf_angle_end   : Angle of the FGF permissive zone when t=t_max
        fgf_angle_rate  : Rate of angle change (used in exponent)

    Returns:
        numpy.ndarray: A 2D array matching the shape of the tissue_mask array containing 
                       the normalized source values, clipped between 0 and 1, 
                       and transposed for correct orientation.
    """

    # Get center coordinates of the tissue mask
    centerX, centerY = tissue_mask.Cx, tissue_mask.Cy

    # Update parameters
    theta1 = smooth_function(t, fgf_angle_start_1, fgf_angle_end_1, fgf_angle_rate)
    theta2 = smooth_function(t, fgf_angle_start_2, fgf_angle_end_2, fgf_angle_rate)

    # Use boundary points of tissue mask
    iterations = 1
    boundary_mask = np.array(tissue_mask.boundary_mask)

    # Exclude boundary points at the left edge of the domain
    boundary_mask[0:iterations, :] = 0 # HACK

    # Create coordinate grid
    rows, cols = tissue_mask.mask.shape
    y, x = np.ogrid[:rows, :cols]

    # Offset
    dx = x - centerX
    dy = y - centerY

    # Slice condition
    angles = np.arctan2(dy, dx)
    in_slice = (angles >= theta2) & (angles <= theta1)

    boundary_mask*= in_slice.T # Transpose to get right orientation
    
    return(boundary_mask.astype("float64"))


def compute_hoxd13_zone(timestep, tissue_mask):
    """
    Generate a binary mask for an elliptical slice in a 2D array.

    Parameters:
        hox (np.ndarray): Input 2D array (shape used for mask).
        Ex, Ey (int): Center coordinates of the ellipse.
        Lx, Ly (int): Semi-axes lengths of the ellipse.

    Global variables:
        
        hox_angle_start : Angle of the Hoxd13 permissive zone when t=0
        hox_angle_end   : Angle of the Hoxd13 permissive zone when t=t_max
        hox_angle_rate  : Rate of angle change (used in exponent)

        hox_minR    : Minimum radius for Hoxd13 permissive zone
        hox_maxR    : Maximum radius for Hoxd13 permissive zone
        hox_R_rate  : Rate of change of the Hoxd13 minimum radius over time
        
    Returns:
        np.ndarray: Binary mask (same shape as array), 1 for points inside slice, 0 otherwise.
    """
    
    # Get center coordinates of the tissue mask
    Ex, Ey = tissue_mask.Cx, tissue_mask.Cy

    if tissue_mask.geometry == 'ellipse':
        Rx,Ry = tissue_mask.Lx, tissue_mask.Ly

    elif tissue_mask.geometry == 'paddle':  
        Rx,Ry = tissue_mask.Ly, tissue_mask.Ly
    
    # Update hox parameters
    theta1 = smooth_function(timestep, hox_angle_start_1, hox_angle_end_1, hox_angle_rate)
    theta2 = smooth_function(timestep, hox_angle_start_2, hox_angle_end_2, hox_angle_rate)

    minR  = smooth_function(timestep, hox_maxR, hox_minR, hox_R_rate)
    maxR  = hox_maxR

    # Create coordinate grid
    rows, cols = tissue_mask.mask.shape
    y, x = np.ogrid[:rows, :cols]

    # Translate coordinates to ellipse center
    dx = x - Ex
    dy = y - Ey

    # Calculate angle
    angles = np.arctan2(dy, dx)

    # Ellipse equation
    in_ellipse_max = (dx**2 / Rx**2) + (dy**2 / Ry**2) <= maxR**2
    in_ellipse_min = (dx**2 / Rx**2) + (dy**2 / Ry**2) >= minR**2

    # Slice condition
    in_slice = (angles >= theta2) & (angles <= theta1)

    # Combine all conditions
    mask = in_ellipse_max & in_ellipse_min & in_slice

    # Apply a blur filter to the binary mask to smooth the edges
    mask = ndimage.binary_dilation(mask, iterations = 2)
    mask = ndimage.gaussian_filter(mask.astype(float), sigma=3)
    mask = np.clip(mask, 0, 1)

    # Transpose in return statement to get right orientation
    return(tissue_mask.mask * mask.T)


def growth_bilinear_interpolation(input_matrix, input_mask, output_mask):
    """
    Use bilinear interpolation to grow the input domain to the new size.
    """ 

    minxI, maxxI, minyI, maxyI = input_mask.bounds
    minxO, maxxO, minyO, maxyO = output_mask.bounds
    
    xrangeI = maxxI - minxI
    yrangeI = maxyI - minyI

    xrangeO = maxxO - minxO
    yrangeO = maxyO - minyO

    # No growth, no need to calculate
    if ( xrangeI - xrangeO == 0 ) & ( yrangeI - yrangeO == 0 ):
        return(input_matrix)

    # Extract tissue values
    input_mask_nan = np.where(input_mask.mask == 1, input_mask.mask, np.nan)
    inner_matrix = input_matrix[minxI:maxxI, minyI:maxyI] * input_mask_nan[minxI:maxxI, minyI:maxyI]

    # Map new matrix indices to old matrix indices
    new_x = np.linspace(0, 1, xrangeO)
    new_y = np.linspace(0, 1, yrangeO)
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
    output_matrix[minxO:maxxO, minyO:maxyO] = out_matrix

    output_matrix = np.where(np.isnan(output_matrix), 0, output_matrix)

    return output_matrix


def single_step_growth(input_matrix, input_mask, output_mask, growth_type = 'add'):
    """
    Wrapper function for choosing growth function.
    """

    if np.sum(output_mask.mask) - np.sum(input_mask.mask) < 0:
        print("Warning: Domain is shrinking. Growth functions do not support shrinking domain.")
        sys.exit()

    if growth_type == 'add':
        # Grow by incorporating "neutral" (0-valued)
        output_matrix = input_matrix

    elif growth_type == 'stretch':
        # Grow by "stretching" values of old to the new domain size 
        output_matrix = growth_bilinear_interpolation(input_matrix, input_mask, output_mask)
    
    else:
        print("Undefined growth type defined, choose 'add' or 'stretch':", growth_type)
        print("Defaulting to 'add' growth type.")
        
        output_matrix = input_matrix

    return(output_matrix)


def single_time_step(sox9, bmp, wnt, fgf, hox,       # matrices with variables
                     Dbmp, Dwnt, Dfgf,               # diffusion coefficients
                     k2, k3, k4, k5, k7, k9, delta,  # reaction parameters
                     fgf_decay_rate,                 # FGF gradient decay rate
                     k_HF_bmp, k_HF_wnt,             # FGF / Hoxd13 parameters for BMP and Wnt
                     tissue_mask                     # binary mask representing tissue
                     ):
    
    """
    Solves a single time step of the reaction-diffusion simulation 
    using explicit forward Euler.
    The diffusion operator (Laplacian) is solved with a 5-point stencil.
    See also: 
    https://en.wikipedia.org/wiki/FTCS_scheme
    https://en.wikipedia.org/wiki/Five-point_stencil
    """

    # Get the bounds of the tissue mask to reduce the area of calculations
    minx, maxx, miny, maxy = tissue_mask.bounds
    number_of_flux_neighbours = tissue_mask.mask_shift_sum
    tissue_mask = tissue_mask.mask

    # DIFFUSION OPERATOR
    # Calculate the Laplacian in the domain excluding the boundaries
    # Use a 5-point stencil

    # Create a bigger matrix with one extra row and column on each side
    bigL_bmp = np.zeros((bmp.shape[0] + 2, bmp.shape[1] + 2))
    bigL_wnt = np.zeros((wnt.shape[0] + 2, wnt.shape[1] + 2))
    bigL_fgf = np.zeros((fgf.shape[0] + 2, fgf.shape[1] + 2))

    # Fill inner values - boundaries are taken care of by mask
    bigL_bmp[1:-1, 1:-1] = bmp
    bigL_wnt[1:-1, 1:-1] = wnt
    bigL_fgf[1:-1, 1:-1] = fgf


    # Calculate the Laplacian

    # Create empty matrices for laplacians
    laplacian_bmp = np.zeros_like(bmp)
    laplacian_wnt = np.zeros_like(wnt)
    laplacian_fgf = np.zeros_like(fgf)

    # all indices shifted by one to account for extra row and column
    minX = minx + 1
    maxX = maxx + 1   
    minY = miny + 1
    maxY = maxy + 1

    # Sum over neighbours and subtract number_of_neighbours*center_value to account for no-flux boundary
    laplacian_bmp[minx:maxx, miny:maxy] = (bigL_bmp[minX+1: maxX+1, minY:maxY] + 
                                           bigL_bmp[minX-1: maxX-1, minY:maxY] + 
                                           bigL_bmp[minX  : maxX  , minY+1:maxY+1] + 
                                           bigL_bmp[minX  : maxX  , minY-1:maxY-1] +
                                           -  bigL_bmp[minX:maxX, minY:maxY]*number_of_flux_neighbours[minx:maxx, miny:maxy] )
    
    laplacian_wnt[minx:maxx, miny:maxy] = (bigL_wnt[minX+1: maxX+1, minY:maxY] + 
                                           bigL_wnt[minX-1: maxX-1, minY:maxY] + 
                                           bigL_wnt[minX  : maxX  , minY+1:maxY+1] + 
                                           bigL_wnt[minX  : maxX  , minY-1:maxY-1] +
                                           - bigL_wnt[minX:maxX, minY:maxY]*number_of_flux_neighbours[minx:maxx, miny:maxy])

    laplacian_fgf[minx:maxx, miny:maxy] = (bigL_fgf[minX+1: maxX+1, minY:maxY] + 
                                           bigL_fgf[minX-1: maxX-1, minY:maxY] + 
                                           bigL_fgf[minX  : maxX  , minY+1:maxY+1] + 
                                           bigL_fgf[minX  : maxX  , minY-1:maxY-1] +
                                           - bigL_fgf[minX:maxX, minY:maxY]*number_of_flux_neighbours[minx:maxx, miny:maxy])
    
    
    # REACTIONS + DIFFUSION
    
    # Calculate the change in variables in step dt
    dsox9 = np.zeros_like(sox9)
    dbmp  = np.zeros_like(bmp)
    dwnt  = np.zeros_like(wnt)
    dfgf  = np.zeros_like(fgf)

    # Break equations up into multiple lines for readability

    # Based on Supplementary Equation 33, simplified to not have production terms and steady state offset.
    # Scaling parameters lambda an gamma are also not considered, as including those would require
    # a much finer time stepping (or a more robust solver).

    # Equation for change in sox9
    dsox9[minx:maxx, miny:maxy] = dt*(+ k2 * bmp[minx:maxx, miny:maxy] 
                                  - k3 * wnt[minx:maxx, miny:maxy]
                                  - sox9[minx:maxx, miny:maxy]**3 
                                  + delta * sox9[minx:maxx, miny:maxy]**2)

    # Equation for change in bmp
    dbmp[minx:maxx, miny:maxy] = dt*((Dbmp / (dx * dx)) * laplacian_bmp[minx:maxx, miny:maxy]
                                 -(k4 - k_HF_bmp*fgf[minx:maxx, miny:maxy]*hox[minx:maxx, miny:maxy])*sox9[minx:maxx, miny:maxy]
                                 - k5 * bmp[minx:maxx, miny:maxy] )

    # Equation for change in wnt    
    dwnt[minx:maxx, miny:maxy] = dt*((Dwnt / (dx * dx)) * laplacian_wnt[minx:maxx, miny:maxy]
                                 - (k7 + k_HF_wnt*fgf[minx:maxx, miny:maxy]*hox[minx:maxx, miny:maxy])*sox9[minx:maxx, miny:maxy]
                                 - k9 * wnt[minx:maxx, miny:maxy] )

    # Equation for change in fgf
    dfgf[minx:maxx, miny:maxy] = dt*((Dfgf / (dx * dx)) * laplacian_fgf[minx:maxx, miny:maxy] +
                                 - fgf_decay_rate*fgf[minx:maxx, miny:maxy] ) 
    
    # Apply changes to variables
    sox9[minx:maxx, miny:maxy] += dsox9[minx:maxx, miny:maxy]* tissue_mask[minx:maxx, miny:maxy]
    bmp[minx:maxx, miny:maxy]  += dbmp[ minx:maxx, miny:maxy]* tissue_mask[minx:maxx, miny:maxy]
    wnt[minx:maxx, miny:maxy]  += dwnt[ minx:maxx, miny:maxy]* tissue_mask[minx:maxx, miny:maxy]
    fgf[minx:maxx, miny:maxy]  += dfgf[ minx:maxx, miny:maxy]* tissue_mask[minx:maxx, miny:maxy]
    
    return(sox9, bmp, wnt, fgf, hox)


##################
# EXECUTION CODE #
##################
# Check if the simulation parameters respect the FTCS stability criterion.
# See: https://en.wikipedia.org/wiki/FTCS_scheme
if (Dbmp * dt / dx**2) >= 0.25:
    raise ValueError("Stability criterion not met, current value is {}".format(Dbmp * dt / dx**2)) 
else:
    print("Stability criterion met, current value is {}".format(Dbmp * dt / dx**2))


### Define initial condition ###

# Define the initial tissue domain with non-zero values
if geometry == 'ellipse':
    tissue_mask_after = EllipseMask(geometry, Lx0, Ly0, nx, ny, ellipse_center_x, ellipse_center_y)

elif geometry == 'paddle':
    tissue_mask_after = PaddleMask(geometry, Lx0, Ly0, nx, ny)

# Initialize a random value to the tissue domain
sox9 = 0.01 * np.random.randn(nx, ny) * tissue_mask_after.mask
bmp  = 0.01 * np.random.randn(nx, ny) * tissue_mask_after.mask
wnt  = 0.01 * np.random.randn(nx, ny) * tissue_mask_after.mask

fgf  = compute_FGF_source_on_mask(0, tissue_mask_after)
hox = np.zeros((nx, ny)) 

# Set up plots
# create a separate axis for each variable
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 6))    

# Define the min and max values for each colormap
sox9_vmin, sox9_vmax = -1.5, 1.5  
bmp_vmin, bmp_vmax = -7.5, 7.5
wnt_vmin, wnt_vmax = -10, 10 

# Define custom colors for plots
colors = [(0, 0, 0), (1, 0.2, 0.2)] 
red_dark_cmap = LinearSegmentedColormap.from_list('black_to_red', colors, N=256)
colors = [(0, 0, 0), (0, 1, 0)] 
green_dark_cmap = LinearSegmentedColormap.from_list('black_to_green', colors, N=256)
colors = [(0, 0, 0), (0, 1, 1)] 
blue_dark_cmap = LinearSegmentedColormap.from_list('black_to_blue', colors, N=256)


### Simulation loop ###

# A simulation step consists of three events:
#   1. A bigger domain is defined
#   2. A growth function is called that implements growth from the old to the new domain size
#   3. Reaction and diffusion is simulated on the new domain
# The output is plotted every 100 simulation steps.

# Find for which timestep the domain might change size for X and Y to update mask when necessary
growth_timesteps_x = [t for t in range(time_steps) if get_domain_size_horizontal(t) != get_domain_size_horizontal(t-1)]
growth_timesteps_y = [t for t in range(time_steps) if get_domain_size_vertical(t) != get_domain_size_vertical(t-1)]

growth_timesteps = sorted(set(growth_timesteps_x + growth_timesteps_y)) # Combine the lists, make values unique, then sort them

for timestep in range(time_steps):
    
    # Check if it is necessary to do calculations of steps 1 and 2
    if timestep in growth_timesteps:

        # Step 1 -- define the new domain size
        tissue_mask_before = tissue_mask_after.copy()
        
        # update mask to new size
        tissue_mask_after.update(get_domain_size_horizontal(timestep), 
                                 get_domain_size_vertical(timestep), 
                                 max_domain_horizontal, 
                                 max_domain_vertical) 
        
        # Step 2 -- grow the tissue
        bmp  = single_step_growth(bmp, tissue_mask_before, tissue_mask_after, growth_type = growth_type)
        sox9 = single_step_growth(sox9, tissue_mask_before, tissue_mask_after, growth_type = growth_type)
        wnt  = single_step_growth(wnt, tissue_mask_before, tissue_mask_after, growth_type = growth_type)
        fgf  = single_step_growth(fgf, tissue_mask_before, tissue_mask_after, growth_type = growth_type)

    else:
        tissue_mask_before = tissue_mask_after  # no change in size, keep old mask.

    # Compute the HoxD13 zone for the current timestep and tissue domain
    hox = compute_hoxd13_zone(timestep, tissue_mask_after)

    # Compute the FGF source for the current timestep and tissue domain
    fgf_source = compute_FGF_source_on_mask(timestep, tissue_mask_after)
    fgf[fgf_source == 1] = 1.0

    # Step 3 -- Simulate reaction and diffusion on the domain
    sox9, bmp, wnt, fgf, hox = single_time_step(sox9, bmp, wnt, fgf, hox,
                                                Dbmp, Dwnt, Dfgf, 
                                                k2, k3, k4, k5, k7, k9, delta, fgf_decay_rate,
                                                k_HF_bmp, k_HF_wnt,
                                                tissue_mask_after)

    ### Plotting ###
    if timestep == 0:  # Initial plot

        plot_mask = np.where(tissue_mask_after.mask == 0, np.nan, tissue_mask_after.mask) 

        current_time = timestep*dt

        sox9_plot = ax1.imshow(np.rot90(sox9*plot_mask), cmap=red_dark_cmap, vmin=sox9_vmin, vmax=sox9_vmax)
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot = ax2.imshow(np.rot90(bmp*plot_mask), cmap=green_dark_cmap, vmin=bmp_vmin, vmax=bmp_vmax)
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot = ax3.imshow(np.rot90(wnt*plot_mask), cmap=blue_dark_cmap, vmin=wnt_vmin, vmax=wnt_vmax)
        ax3.set_title(f'wnt at step {current_time}')

        fgf_plot = ax4.imshow(np.rot90(fgf*plot_mask), cmap='viridis', vmin=0, vmax=1)
        ax4.set_title(f'fgf at step {current_time}')

        hox_plot = ax5.imshow(np.rot90(hox*plot_mask), cmap='plasma', vmin=0, vmax=1)
        ax5.set_title(f'hoxd13 at step {current_time}')
        
        ax6.axis('off')

        plt.pause(0.000001)  # Pause to refresh the plot

    # Update plot every 100 steps    
    elif timestep % 100 == 0:

        plot_mask = np.where(tissue_mask_after.mask == 0, np.nan, tissue_mask_after.mask) 

        current_time = timestep*dt
        
        sox9_plot.set_data(np.rot90(sox9*plot_mask))
        ax1.set_title(f'sox9 at step {current_time}')

        bmp_plot.set_data(np.rot90(bmp*plot_mask))
        ax2.set_title(f'bmp at step {current_time}')

        wnt_plot.set_data(np.rot90(wnt*plot_mask))
        ax3.set_title(f'wnt at step {current_time}')

        fgf_plot.set_data(np.rot90(fgf*plot_mask))
        ax4.set_title(f'fgf at step {current_time}')

        hox_plot.set_data(np.rot90(hox*plot_mask))
        ax5.set_title(f'hoxd13 at step {current_time}')

        plt.pause(0.000001)  # Pause to refresh the plot
    
plt.show()