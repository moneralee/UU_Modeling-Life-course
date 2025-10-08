###
# PRACTICAL 1 | "Every cell for themselves?"
# This is the starting code. Follow the instructions in the practical to complete the code. 
# If you get stuck, you can look at the final code in `foraging_for_resources_final.py`, or ask
# Bram. 
#
# The structure of this code is as follows:
# 1. Imports and parameters
# 2. Simulation class
# 3. Cell class
# 4. Visualisation class (you do not need to change this)
#
###

# 1. IMPORTS AND PARAMETERS
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters for simulation
WORLD_SIZE = 200    # Width / height of the world (size of grid and possible coordinates for cells)
MAX_VELOCITY = 0.3  # Maximum velocity magnitude
MAX_FORCE = 0.3     # Maximum force magnitude
RANDOM_MOVEMENT  = 0.01 # Random movement factor to add some noise to the cell's movement

# Parameters for display
DRAW_ARROW = True  # Draw the arrows showing the velocity direction of the cells
INIT_CELLS = 20 # Initial number of cells in the simulation
DISPLAY_INTERVAL = 1 # Frequency with which the plot is updated (e.g., every 10 timesteps can speed things up)

# 1. MAIN LOOP (using functions and classes defined below)
def main():
    """Main function to set up and run the simulation."""
    # NOTE: The `Visualisation` class is responsible for managing the visualization 
    # of the simulation, including creating plots, updating them, and handling 
    # user interactions like the slider. As this has nothing to do with modeling
    # per se, understanding this code is not necessary, but it can be fun to look
    # at if you are interested. 
    
    num_cells = INIT_CELLS
    sim = Simulation(num_cells) 

    plt.ion()
    vis = Visualisation(sim)

    def update_cells(val):
        sim.initialise_cells(int(vis.slider.val))
        vis.redraw_plot(sim)
        
    # Connect the slider to the update function
    vis.slider.on_changed(update_cells)

    # Run simulation
    for t in range(1, 10000):
        
        sim.simulate_step()
        
        if(t % DISPLAY_INTERVAL == 0):
            # As long as only cells move, update only positions and timestamp
            vis.update_plot(sim) 
            vis.ax.set_title(f"Timestep: {t}")
            vis.fig.canvas.draw_idle()
            plt.pause(10e-20)        
        if(sim.redraw):
            # When more has changes (e.g. number of cells or target position), redraw the plot
            vis.redraw_plot(sim) 
            sim.redraw = False # Make sure it doesn't keep redrawing if not necessary
        

    # Keep the final plot open
    plt.ioff()
    # plt.show()



# 2. SIMULATION CLASS
class Simulation:
    """Manages the grid, cells, target, and simulation logic."""
    def __init__(self, num_cells):
        # Initialise a grid for the simulation
        self.grid = np.zeros((WORLD_SIZE, WORLD_SIZE))  # Initialise an empty grid
        self.fill_grid(self.grid, 0, 0, 0, 0)           # Fill grid with values (currently just 1s)
        # Initialise a population of cells
        self.cells = []
        self.initialise_cells(num_cells)
        # Place a 'target' in the middle
        self.target_position = [WORLD_SIZE/2, WORLD_SIZE/2]  # Initial target position at the center
        # A flag to only rebuild the plot when necessary (e.g. when the number of cells changes)
        self.redraw = False

    def simulate_step(self):
        """Simulate one timestep of the simulation."""
        for cell in self.cells:
            # Actions taken by each cell. Most of them are still undefined, so you can implement them yourself.
            self.move_towards_dot(cell)  
            if self.check_target_reached(cell):
                print(f"Target reached!")
                self.reproduce_cell(cell)
                self.redraw = True
            
            #self.avoid_collision(cell)
            #self.stick_to_close(cell)
            #self.find_peak(cell)

            # Apply forces and update position
            cell.apply_forces()
            cell.update_position()

            # Limit velocity to the maximum allowed
            cell.vx = np.clip(cell.vx, -MAX_VELOCITY, MAX_VELOCITY)
            cell.vy = np.clip(cell.vy, -MAX_VELOCITY, MAX_VELOCITY)

    def initialise_cells(self, num_cells):
        """Initialise the cells with random positions and velocities."""
        self.cells = []
        for _ in range(num_cells):
            x = np.random.uniform(0, WORLD_SIZE)
            y = np.random.uniform(0, WORLD_SIZE)
            vx = np.random.uniform(-1, 1)
            vy = np.random.uniform(-1, 1)
            self.cells.append(Cell(x, y, vx, vy))

    def fill_grid(self, grid, mean_x, mean_y, std_dev, noise=0):
        """
        Write a function that takes the 2D grid and fills it with values representing 
        a Gaussian (normal) distribution centered at (mean_x, mean_y). See
        if you can use the 'noise' argument to randomise the gaussian distribution a bit.
        
        Hint: e^{-x^2} yields a bell curve centered around 0. 
        
        """
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                grid[i, j] = 1 # This is 1 in the example, but should be a Gaussian distribution

        # Normalize the grid to keep the total resource concentration the same
        self.grid = grid
    
    def find_peak(self, cell):
        """Make the cell move towards the peak of the resource gradient with a random walk."""
        # Convert cell position to grid indices, as well as the previous position
        grid_x = int(cell.x) % WORLD_SIZE
        grid_y = int(cell.y) % WORLD_SIZE
        next_x = (int(cell.x + 30*cell.vx) + WORLD_SIZE) % WORLD_SIZE 
        next_y = (int(cell.y + 30*cell.vy) + WORLD_SIZE) % WORLD_SIZE 
         
    
    def avoid_collision(self, cell):
        """Implement a simple collision avoidance mechanism. You can do so by
        checking if this individual overlaps with another individual, and if so,
        applying a repulsion force to the individual apposing the overlapping
        direction."""
        for other_cell in self.cells:
            if other_cell is not cell:
                # Calculate the distance between the two cells
                dx = cell.x - other_cell.x
                dy = cell.y - other_cell.y
                distance = np.sqrt(dx**2 + dy**2)
                
                    
    def stick_to_close(self, cell):
        """Implement an attraction to cells that are nearby (but not overlapping)"""
        for other_cell in self.cells:
            if other_cell is not cell:
                # Calculate the distance between the two cells
                dx = cell.x - other_cell.x
                dy = cell.y - other_cell.y
                distance = np.sqrt(dx**2 + dy**2)

    
    def move_towards_dot(self, cell):
        """
        Write your own function that applies forces in the direction of the dot.
        Try to think of a way to apply the same force to every cell irrespective
        of the distance to the dot, such that the cells move towards the dot at 
        the same speed. 
        
        To get you started, the function already calculates dx and dy, which are
        the distances to the target position in the x and y direction, respectively.
        """
        # Calculate dx and dy
        dx = self.target_position[0] - cell.x
        dy = self.target_position[1] - cell.y
        
    
    def check_target_reached(self, cell):
        """
        Write your own function that checks if this cell has reached the target position.
        You can do this by calculating the distance between the cell and the target.
        If the distance is smaller than a certain threshold (e.g., 3 units), return True.
        Otherwise, return False.
        """
        
        return(False)  # Dummy 'return' value. 
    
    def reproduce_cell(self, cell):
        """
        Write your own function that reproduces this cell. Think
        about what it should inherit, and what it should *not* inherit. 
        
        To keep the number of cell constant, you can first throw away a random cell.
        """
        # Reproduce: Create a new cell with the same properties as the current cell
        return(False) # Dummy 'return' value.

        
        
# 3. CELL CLASS
class Cell:
    """Represents an individual cell in the simulation."""
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.ax = 0
        self.ay = 0
        self.stickiness = 0.01 # Initial stickiness, can be adjusted later
        
    def update_position(self):
        """Update the cell's position based on its velocity."""
        self.x = (self.x + self.vx ) % WORLD_SIZE  # Wrap around the world
        self.y = (self.y + self.vy ) % WORLD_SIZE  # Wrap around the world

    def apply_forces(self):
        """Apply a force to the cell, updating its velocity."""
        self.ax = np.clip(self.ax, -MAX_FORCE, MAX_FORCE)
        self.ay = np.clip(self.ay, -MAX_FORCE, MAX_FORCE)
        self.vx += self.ax + RANDOM_MOVEMENT * np.random.uniform(-1, 1)
        self.vy += self.ay + RANDOM_MOVEMENT * np.random.uniform(-1, 1)
        # Apply drag to slow down the cell naturally
        self.ax = 0
        self.ay = 0
        


# Visualisation class for showing the individuals and the grid. For the practical, you do not need to change this. 
class Visualisation:    
    def __init__(self, sim):
        fig, ax = plt.subplots(figsize=(6, 6))
        self.cell_x = [cell.x for cell in sim.cells]
        self.cell_y = [cell.y for cell in sim.cells]
        self.cell_vx = np.array([cell.vx for cell in sim.cells])
        self.cell_vy = np.array([cell.vy for cell in sim.cells])
        self.cell_stickiness = np.array([cell.stickiness for cell in sim.cells])
        # Colour cells by stickiness using inferno colormap
        self.cell_scatter = ax.scatter(self.cell_x, self.cell_y, c=self.cell_stickiness, cmap='inferno', s=50, edgecolor='white', vmin=0, vmax=1)
        if(DRAW_ARROW): self.cell_quiver = ax.quiver(self.cell_x, self.cell_y, self.cell_vx * 0.5, self.cell_vy * 0.5, angles='xy', scale_units='xy', scale=0.02, color='white')
        plt.subplots_adjust(bottom=0.2)

        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Timestep: 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        target_point=ax.scatter(sim.target_position[0], sim.target_position[1], c='purple', s=100, edgecolor='red')
        grid_im=ax.imshow(sim.grid.T, extent=(0, WORLD_SIZE, 0, WORLD_SIZE), origin='lower', cmap='viridis', alpha=1.0)

        self.fig = fig
        self.ax = ax
        self.target_point = target_point
        self.grid_im = grid_im

        # Add a slider for selecting the number of cells
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.slider = Slider(ax_slider, 'Cells', 1, 1000, valinit=len(sim.cells), valstep=1)

    def update_cell_positions(self, sim):
        """Update the positions of the cells in the visualisation."""
        self.cell_x = [cell.x for cell in sim.cells]
        self.cell_y = [cell.y for cell in sim.cells]
        self.cell_vx = np.array([cell.vx for cell in sim.cells])
        self.cell_vy = np.array([cell.vy for cell in sim.cells])
        self.cell_stickiness = np.array([cell.stickiness for cell in sim.cells])
    
    def update_plot(self, sim):
        self.update_cell_positions(sim)
        self.cell_scatter.set_offsets(np.c_[self.cell_x,self.cell_y])
        self.cell_scatter.set_array(self.cell_stickiness)
        if(DRAW_ARROW): 
            self.cell_quiver.set_offsets(np.c_[self.cell_x, self.cell_y])
            self.cell_quiver.set_UVC(self.cell_vx * 0.5, self.cell_vy * 0.5)        

    def redraw_plot(self, sim):
        self.update_cell_positions(sim)
        cell_scatter_new = self.ax.scatter(self.cell_x, self.cell_y, c=self.cell_stickiness, cmap='inferno', s=50, edgecolor='white', vmin=0, vmax=1)
        if(DRAW_ARROW): 
            cell_quiver_new = self.ax.quiver(self.cell_x, self.cell_y, self.cell_vx * 0.15, self.cell_vy * 0.15, angles='xy', scale_units='xy', scale=0.02, color='white')
            self.cell_quiver.remove()
            self.cell_quiver = cell_quiver_new
        self.cell_scatter.remove()
        self.fig.canvas.draw_idle()
        self.cell_scatter = cell_scatter_new
        self.grid_im.remove()
        self.grid_im = self.ax.imshow(sim.grid.T, extent=(0, WORLD_SIZE, 0, WORLD_SIZE), origin='lower', cmap='viridis', alpha=1.0)
        self.target_point.remove()
        self.target_point=self.ax.scatter(sim.target_position[0], sim.target_position[1], c='purple', s=100, edgecolor='red')
        plt.pause(10e-20)
            
            
# 4. Execute the main loop
if __name__ == "__main__":
    # with cProfile.Profile() as pr:
        main()
        # pr.print_stats()