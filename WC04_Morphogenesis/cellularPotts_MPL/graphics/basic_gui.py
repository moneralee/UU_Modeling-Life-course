"""
Optimized Matplotlib-based visualization of CPM simulation
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve


class Visualization:
    def __init__(self, params, cpm):
        self.params = params
        self.cpm = cpm
        self.t_tot = params["t_tot"]
        self.freq = int(params["freq"])
        self.scale = int(params["scale"])
        self.MCS = 0
        self.initialization = False
        
        # Precompute color table and ID mapping
        self.set_color_table()
        
        # Precompute scaling indices for faster repeat operation
        self.h, self.w = self.cpm.sigma_field.shape
        self.scaled_h = self.h * self.scale
        self.scaled_w = self.w * self.scale
        
        # Pre-allocate arrays
        self.scaled_sigma = np.zeros((self.scaled_h, self.scaled_w), dtype=self.cpm.sigma_field.dtype)
        self.sigma_color = np.zeros((self.scaled_h, self.scaled_w), dtype=np.uint8)
        
        # Precompute boundary detection kernel (more efficient than multiple rolls)
        self.boundary_kernel = np.array([[0, 1, 0],
                                         [1, 0, 1],
                                         [0, 1, 0]], dtype=np.int8)
        
        # Temporary array for boundary detection
        self.boundary_conv = np.zeros((self.scaled_h, self.scaled_w), dtype=np.int32)
        
        # Create display
        self.fig, self.ax = plt.subplots()
        self.ax.axis("off")
        self.ax.set_title("CPM Visualization", fontsize=12)
        
        # Initial image
        self.im = self.ax.imshow(self.sigma_color,
                                 cmap=self.cmap,
                                 vmin=0,
                                 vmax=len(self.cmap.colors) - 1,
                                 interpolation="none",
                                 animated=True)
    
    def set_color_table(self):
        """Build a matplotlib colormap instead of RGB array"""
        c_ids = self.cpm.cell_ids
        ctype = self.cpm.c_types
        
        # Map cell IDs to color indices
        max_id = np.max(c_ids)
        self.id_to_color_index = np.zeros(max_id + 1, dtype=np.uint8)
        self.id_to_color_index[c_ids] = np.asarray(ctype, dtype=np.uint8)
        
        # Build colormap (normalize RGB to 0–1)
        colors = np.array(list(self.params["colors"].values()), dtype=np.float32) / 255.0
        boundary_color = np.array([[0, 0, 0]], dtype=np.float32)  # black for cell edges
        all_colors = np.vstack([colors, boundary_color])
        self.cmap = ListedColormap(all_colors)
        self.boundary_idx = len(all_colors) - 1
    
    def compute_display_field(self):
        """Compute scaled and color-indexed image with optimized operations."""
        sigma = self.cpm.sigma_field
        
        # Scaling using numpy repeat with pre-allocated array
        if self.scale == 1:
            self.scaled_sigma = sigma
        else:
            # Use direct indexing for scaling (faster than repeat for small scales)
            if self.scale <= 4:
                for i in range(self.scale):
                    for j in range(self.scale):
                        self.scaled_sigma[i::self.scale, j::self.scale] = sigma
            else:
                np.repeat(np.repeat(sigma, self.scale, axis=0), self.scale, axis=1, 
                         out=self.scaled_sigma)
        
        # Map sigma to color index using lookup table
        np.take(self.id_to_color_index, self.scaled_sigma, out=self.sigma_color)
        
        # Boundary detection using convolution
        boundaries = self.get_perimeter_elements_fast(self.scaled_sigma)
        self.sigma_color[boundaries] = self.boundary_idx
        
        return self.sigma_color
    
    def get_perimeter_elements_fast(self, field):
        """Boundary detection using scipy convolution."""
        # Convolve with neighbor kernel
        convolve(field, self.boundary_kernel, output=self.boundary_conv, 
                mode='constant', cval=0)
        
        # Find boundaries: positions where convolved value differs from 4*field
        # (meaning at least one neighbor is different)
        boundaries = self.boundary_conv != (field * 4)
        
        return boundaries
    
    def step(self, frame):
        """Perform simulation step(s) and update every freq frames."""
        if self.initialization:
            self.cpm.initialize(J0=-8, n_initialise_steps=self.params["t_ini"])
            self.initialization = False
        else:
            # Perform simulation step
            self.cpm.simulate()
            self.MCS += 1
        
        # Print progress less frequently
        if self.MCS % 100 == 0:
            print(f"MCS {self.MCS}")
        
        # Only update display occasionally
        if self.MCS % self.freq == 0:
            self.compute_display_field()
            self.im.set_data(self.sigma_color)
            return [self.im]
        return []
    
    def run(self):
        """Start the matplotlib animation loop."""
        anim = animation.FuncAnimation(
            self.fig,
            self.step,
            frames=self.t_tot,
            interval=1,
            blit=True,
            repeat=False,
            cache_frame_data=False  # Reduce memory overhead
        )
        plt.show()
        print("Done!")