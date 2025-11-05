"""
Barebones visualization of CPM simulation
"""

#################################

import time # TODO: remove once done testing
start_time = time.time()

# Installed modules
import numpy as np

#PyQT5 for visualization
from PyQt5 import QtWidgets, QtCore, QtGui

#################################

class Visualization(QtWidgets.QMainWindow):

    def __init__(self, params, cpm, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.label = QtWidgets.QLabel()
        self.setCentralWidget(self.label)
        self.setWindowTitle("CPM Visualization")

        self.params = params

        self.t        = np.arange(self.params["t_tot"])
        self.t_update = self.t[::int(self.params["freq"])]

        self.update_i = 0

        self.MCS = 0
        self.cpm = cpm
        self.initialization = True
        self.set_color_table()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.step_and_update)
        self.timer.start(1)  # Update every t ms
    

    def set_color_table(self):
        # Create colormap

        c_ids = self.cpm.cell_ids # unique non-zero sigma values
        ctype = self.cpm.c_types  # list of cell types, maps 1-to-1 to c_ids
        
        # Create a mapping from cell ID to color index
        # id_to_color_index is a 1D array where the index represents a cell ID, 
        # and the value at that index represents the corresponding color index
        self.id_to_color_index = np.zeros(np.max(c_ids) + 1, dtype=np.uint8)
        self.id_to_color_index[1:] = np.asarray(ctype) # First element (0) is background index
        
        # color table for QImage
        self.colortable = []
        for _, v in self.params["colors"].items():
            self.colortable += [QtGui.qRgb(*v)]
        

    def update_image(self):
        
        sigma = self.cpm.sigma_field  # cpm.sigma_field is a 2D array

        # Scale up the array for visualization
        res = self.params["scale"]
        I_scale = np.repeat(np.repeat(sigma, res, axis=0), res, axis=1)

        # Set the cell boundaries to a special value to color them differently
        boundaries = self.cpm.get_perimeter_elements(I_scale)
        
        # Map scaled sigma field to color indices
        sigma_color_indices = self.id_to_color_index[I_scale]
        sigma_color_indices[boundaries] = len(self.colortable)-1 # last index of colortable

        # Convert to 8-bit for QImage
        sigma8 = sigma_color_indices.astype(np.uint8)

        h, w = sigma8.shape
        img = QtGui.QImage(sigma8.data, w, h, w, QtGui.QImage.Format.Format_Indexed8)
        img.setColorTable(self.colortable)
        pix = QtGui.QPixmap.fromImage(img)
        self.label.setPixmap(pix)


    def step_and_update(self):
        
        # Initialize with a J-matrix that favors cell separation
        if self.initialization:

            self.cpm.initialize(J0=-8, n_initialise_steps = self.params["t_ini"])
            self.initialization = False

        # Simulate normally
        else:
            self.MCS += 1

            if self.MCS == 1:
                s_time = time.time() # TODO - remove when done testing

            self.cpm.simulate()

            if self.MCS == 1:
                end_time = time.time()
                elapsed_time = end_time - s_time
                print(f"Elapsed time after MCS {self.MCS}: {elapsed_time:.4f} seconds")
        
        # Update visualization
        if self.MCS in self.t_update:
            self.update_image()
            self.update_i += 1

        if self.MCS % 100 == 0:
            print(f"MCS {self.MCS}")
        
        # Reached final simulation step - finish
        if self.MCS >= self.params["t_tot"]:
            self.timer.stop()
            print("Progress: 100.0 %")            
            print("Done!")

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Elapsed time TOTAL: {elapsed_time:.4f} seconds")