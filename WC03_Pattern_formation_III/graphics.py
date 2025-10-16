import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
from numpy import ceil
from operator import attrgetter

class TissuePlot:
    def __init__(self, tissue, time_visualization, dt,n_axis=1):
        self.tissue = tissue
        self.time_visualization = time_visualization
        self.dt = dt
        self.max_x_display = self.tissue.xmax_tissue * 16
        # Plotting functions
        self.fig, self.ax = plt.subplots(nrows=n_axis,ncols=1,figsize=(10, 4*n_axis),squeeze=False, layout='compressed')
        self.fig.suptitle('Tissue development at time = 0', fontsize=16)
        self.patches=[None]*n_axis
        self.cmaps=[None]*n_axis
        self.normals=[None]*n_axis
        self.cbs=[None]*n_axis
        self.caxs=[None]*n_axis
        self.labels=[None]*n_axis
        self.update_functions_arguments=[None]*n_axis
        self.attribute_bounds = [[0,0] for n in range(n_axis)]



    def initialize_axis_cell_data(self, axis_index=0, colormap='viridis', attribute='fgf', update_direction={'min_down':0.8,'max_up':1.1,'max_down':0.75,'min_up':1.1}, label=None):
        # Find the desired min and max values for the color map
        data, bounds,_ = self.update_attribute_levels(axis_index, attribute=attribute, direction=update_direction)
        self.normals[axis_index] = plt.Normalize(*bounds)
        self.update_functions_arguments[axis_index] = (axis_index, attribute, update_direction)
        
        self.cmaps[axis_index] = plt.get_cmap(colormap)

        # create rectangles for each cell
        # and add them to the axis
        self.patches[axis_index] = []
        for index, cell in enumerate(self.tissue.cells):
            rect = patches.Rectangle(
                (cell.xmin, 0),  # (x, y)
                cell.xmax - cell.xmin,  # width
                cell.xheight,           # height
                facecolor=self.cmaps[axis_index](self.normals[axis_index](data[index])),
                edgecolor=to_rgba('black', 0.25))
            self.ax[axis_index,0].add_patch(rect)
            self.patches[axis_index].append(rect)
        
        # Set x and y limits and labels
        self.ax[axis_index,0].set_xlim(0, self.max_x_display)
        self.ax[axis_index,0].set_ylim(0, self.tissue.doubling_threshold * 1.01)
        self.ax[axis_index,0].set_xlabel('Cell Position')
        self.ax[axis_index,0].set_ylabel('Cell Height')

        # Create a colorbar
        self.labels[axis_index] = label
        self.caxs[axis_index], _ = cbar.make_axes(self.ax[axis_index,0])
        self.cbs[axis_index] = cbar.ColorbarBase(self.caxs[axis_index], cmap=self.cmaps[axis_index], norm=self.normals[axis_index], label=label)
        if axis_index == len(self.ax)-1:
            plt.pause(0.01)


    def update_plot_cell_data(self, t, axis_index=0):
        # update colorbar when update function returns True
        data, bounds,updated = self.update_attribute_levels(*self.update_functions_arguments[axis_index])      
        if updated:
            self.normals[axis_index] = plt.Normalize(*bounds)
            self.cbs[axis_index] = cbar.ColorbarBase(self.caxs[axis_index], cmap=self.cmaps[axis_index], norm=self.normals[axis_index], label=self.labels[axis_index])
        
        # Update existing rectangles or add new ones if needed
        for index, cell in enumerate(self.tissue.cells):
            if index < len(self.patches[axis_index]):
                rect = self.patches[axis_index][index]
                if index >= self.tissue.num_cells-ceil(2*(self.time_visualization*self.tissue.growth_rate)/self.tissue.doubling_threshold)-1: # only update the last divided cells size to account for growth
                    rect.set_xy((cell.xmin, 0))
                    rect.set_width(cell.xmax - cell.xmin)
                    rect.set_height(cell.xheight)
                rect.set_facecolor(self.cmaps[axis_index](self.normals[axis_index](data[index])))
            else:
                # Add new rectangle if new cell was added
                rect = patches.Rectangle(
                    (cell.xmin, 0),
                    cell.xmax - cell.xmin,
                    cell.xheight,
                    facecolor=self.cmaps[axis_index](self.normals[axis_index](data[index])),
                    edgecolor=to_rgba('black',0.25)
                )
                self.ax[axis_index,0].add_patch(rect)
                self.patches[axis_index].append(rect)
        self.ax[axis_index,0].set_xlim(0, self.max_x_display)
        self.ax[axis_index,0].set_ylim(0, self.tissue.doubling_threshold * 1.01)
        self.fig.suptitle(f'Tissue development at time = {t * self.dt / 3600:.2f} hours', fontsize=16)
        self.ax[axis_index,0].set_title(f'{self.labels[axis_index]}')
        if axis_index == len(self.ax)-1:
            plt.pause(0.01)

    def update_xmax_display(self):
        # Update the maximum x-axis limit for display purposes
        if self.tissue.xmax_tissue >= self.max_x_display:
            self.max_x_display = self.tissue.xmax_tissue * 2       


    def update_attribute_levels(self, axis, attribute='fgf', direction={'min_down':1,'max_up':1,'max_down':1,'min_up':1}):
        # Update the maximum level of any attribute in the tissue
        if self.tissue.cells: 
            # get the attribute levels from the cells
            if '[' in attribute and attribute.endswith(']'):    # If the attribute contains an index':
                attr, index = attribute[:-1].split('[')
                index= int(index.strip(']'))
                attribute_levels= [attrgetter(attr)(cell)[index] for cell in self.tissue.cells]
            else:
                attribute_levels= [attrgetter(attribute)(cell) for cell in self.tissue.cells]
            
            
            # Update the attribute bounds based on the levels
            new_min_max = (min(attribute_levels), max(attribute_levels))
            update = False
            if 'max_up' in direction and new_min_max[1]>self.attribute_bounds[axis][1]:
                # Increase the max attribute levels if they have increased
                self.attribute_bounds[axis][1] = max(new_min_max[1], direction['max_up']*self.attribute_bounds[axis][1])
                update = True
            if 'max_down' in direction and new_min_max[1] < direction['max_down']*self.attribute_bounds[axis][1]:
                # Update the max attribute levels if they have dropped significantly
                self.attribute_bounds[axis][1] = new_min_max[1]
                update = True
            if 'min_down' in direction and new_min_max[0] < self.attribute_bounds[axis][0]:
                # Update the min attribute levels it has dropped 
                self.attribute_bounds[axis][0] = min(new_min_max[0], direction['min_down']*self.attribute_bounds[axis][0])
                update = True
            if 'min_up' in direction and new_min_max[0] > direction['min_up']*self.attribute_bounds[axis][0]:
                # Update the min attribute levels if they have increased
                self.attribute_bounds[axis][0] = new_min_max[0]
                update = True
            return(attribute_levels,self.attribute_bounds[axis],update)        

    def save_plot(self, filename):
        # Save the current plot to a file
        self.fig.savefig(filename)
        print(f"Plot saved as {filename}")