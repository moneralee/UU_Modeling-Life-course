###
# PRACTICAL 1 | "Every cell for themselves?"
# Things in this model that you have tried to implement yourself:
# 1. Implement collision avoidance
# 2. Implement reproduction
# 3. Implement a Gaussian grid
# 4. Implement "run and tumble"
# 5. Add noise to Gaussian, what happens?
# 5. Modify collision into STICKING (a little finicky)
# 6. Try it out with 500 cells... 
###

###
# PRACTICAL 1 | PLENARY DISCUSSION
# What else was discussed in the plenary?
# 1. Why are grids so popular in modelling?
# 2. Tessellation of space
# 3. Automatic tessellation of space: quad tree
# 4. In the full model (javascript/Cacatoo), a quad tree is present, impacting performance
###

# 1. IMPORTS AND PARAMETERS
# Libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Parameters for simulation
WORLD_SIZE = 200  # Width / height of the world (size of grid and possible coordinates for cells)
MAX_VELOCITY = 0.3  # Maximum velocity magnitude
MAX_FORCE = 0.3  # Maximum force magnitude
DRAG_COEFFICIENT = 0.01  # Friction to slow down the cell naturally
RANDOM_MOVEMENT  = 0.01 # Random movement factor to add some noise to the cell's movement
CELL_STICKINESS_LOW = 0.0 # Minimal stickiness of cells in population
CELL_STICKINESS_HIGH = 0.10 # Maximal stickiness of cells in population
# Parameters for display
DRAW_ARROW = False  # Draw the arrows showing the velocity direction of the cells
NOISE = 2 # Noise factor for the Gaussian grid (noise amount is raised to the power of this value)
INIT_CELLS = 64 # Initial number of cells in the simulation
SEASON_DURATION = 1000 # Duration of a season, after which the Gaussian grid is regenerated
DISPLAY_INTERVAL = 5

# 1. MAIN LOOP (using functions and classes defined below)
def main():
    """Main function to set up and run the simulation."""
    # Initialise simulation and its # The `visualis` variable in the code snippet provided is actually
    # a misspelling of the correct variable name `vis`, which stands
    # for the `Visualisation` class instance. The `Visualisation`
    # class is responsible for managing the visualization of the
    # simulation, including creating plots, updating them, and
    # handling user interactions like the slider.
    
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
            vis.update_plot(sim)
            vis.ax.set_title(f"Timestep: {t}")
            vis.fig.canvas.draw_idle()
            plt.pause(10e-20)
        if(t % SEASON_DURATION==0):
            sim.fill_grid(sim.grid, 0.2+np.random.uniform(0,0.6), 0.2+np.random.uniform(0,0.6), 0.1, NOISE)
            vis.redraw_plot(sim)# Create Gaussian grid
        # Update title and redraw the plot

    # Keep the final plot open
    plt.ioff()
    # plt.show()



# 2. SIMULATION CLASS
class Simulation:
    """Manages the grid, cells, target, and simulation logic."""
    def __init__(self, num_cells):
        self.grid = np.zeros((WORLD_SIZE, WORLD_SIZE))  # Initialise an empty grid
        self.cells = []
        self.target_position = [WORLD_SIZE/3, WORLD_SIZE/3]  # Initial target position at the center
        self.target_position = [-1,-1]
        self.fill_grid(self.grid, 0.5, 0.5, 0.1, NOISE)  # Create Gaussian grid
        self.initialise_cells(num_cells)

    def simulate_step(self):
        """Simulate one timestep of the simulation."""
        for cell in self.cells:
            # Actions taken by each cell. Most of them are still undefined, so you can implement them yourself.
            #self.move_towards_dot(cell)  
            #if self.check_target_reached(cell):
            #    print(f"Target reached! New target position: {self.target_position}")
            #    self.reproduce_cell(cell) 
            
            self.avoid_collision(cell)
            self.stick_to_close(cell)
            self.find_peak(cell)
            
            # Apply drag force to acceleration
            cell.ax += -DRAG_COEFFICIENT * cell.vx
            cell.ay += -DRAG_COEFFICIENT * cell.vy

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
        """Creates a Gaussian distribution with noise on the grid."""
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                x = i / (WORLD_SIZE - 1)
                y = j / (WORLD_SIZE - 1)
                distance_squared = (x - mean_x)**2 + (y - mean_y)**2
                grid[i, j] = np.exp(-distance_squared / (2 * std_dev**2)) * np.random.uniform(0.0, 1.0)**noise

        # Normalize the grid to keep the total resource concentration the same
        grid /= np.sum(grid)
        self.grid = grid
    
    def find_peak(self, cell):
        """Make the cell move towards the peak of the resource gradient with a random walk."""
        # Convert cell position to grid indices, as well as the previous position
        grid_x = int(cell.x) % WORLD_SIZE
        grid_y = int(cell.y) % WORLD_SIZE
        next_x = (int(cell.x + 10*cell.vx) + WORLD_SIZE) % WORLD_SIZE 
        next_y = (int(cell.y + 10*cell.vy) + WORLD_SIZE) % WORLD_SIZE 
        # Get the resource value at the cell's position, as well as the previous position
        resource_value = self.grid[grid_x, grid_y]
        resource_next = self.grid[next_x, next_y]
        
        # Check if the cell is moving in the right direction
        if resource_next > resource_value:
            # Moving in the right direction: small random adjustment
            angle = np.random.uniform(-0.1, 0.1)  # Small angle change
        else:
            # Moving in the wrong direction: large random adjustment
            angle = np.random.uniform(-np.pi*1.0, np.pi*1.0)  # Large angle change
        
        # Rotate the velocity vector by the random angle according to trigonometric rotation formulas
        new_vx = cell.vx * np.cos(angle) - cell.vy * np.sin(angle)
        new_vy = cell.vx * np.sin(angle) + cell.vy * np.cos(angle)

        # Update the acceleration with the new velocity vector, such that the cell moves towards the peak
        cell.vx = new_vx
        cell.vy = new_vy
        cell.ax += cell.vx
        cell.ay += cell.vy
         
    
    def avoid_collision(self, cell):
        """Avoidance forces to prevent cells from colliding."""
        for other_cell in self.cells:
            if other_cell is not cell:
                # Calculate the distance between the two cells
                dx = cell.x - other_cell.x
                dy = cell.y - other_cell.y
                distance = np.sqrt(dx**2 + dy**2)

                # If the cells are too close, apply a repulsion force
                if distance < 5.0 and distance > 0:  # Threshold for "too close"
                    # Calculate the repulsion force proportional to the inverse of the distance
                    force_magnitude = (5.0 - distance) / distance
                    cell.ax += force_magnitude * dx  * 100
                    cell.ay += force_magnitude * dy * 100
                    
    def stick_to_close(self, cell):
        """Stick to closeby cells."""
        for other_cell in self.cells:
            if other_cell is not cell:
                # Calculate the distance between the two cells
                dx = cell.x - other_cell.x
                dy = cell.y - other_cell.y
                distance = np.sqrt(dx**2 + dy**2)

                # If the cells are too close, apply a repulsion force
                if distance < 12 and distance > 5:  # Threshold for "close"
                    # Calculate the repulsion force proportional to the inverse of the distance
                    cell.ax -= cell.stickiness * dx *10
                    cell.ay -= cell.stickiness * dy *10
    
    def move_towards_dot(self, cell):
        """Apply forces in the direction of the dot."""
        # Calculate dx and dy
        dx = self.target_position[0] - cell.x
        dy = self.target_position[1] - cell.y
        # Calculate the distance to the target (pythagorean theorem)
        distance = np.sqrt(dx**2 + dy**2)
        
        # Normalize dx and dy 
        dx /= distance
        dy /= distance
        # Apply a small force towards the target
        cell.ax += dx * 0.01
        cell.ay += dy * 0.01
    
    def check_target_reached(self, cell):
        distance_to_target = np.sqrt((cell.x - self.target_position[0])**2 +
                                         (cell.y - self.target_position[1])**2)
        if distance_to_target < 3:
            # Set a new target position
            self.target_position = [np.random.uniform(0, WORLD_SIZE), np.random.uniform(0, WORLD_SIZE)]
            return(True)
        return(False)
    
    def reproduce_cell(self, cell):
        # Reproduce: Create a new cell with the same properties as the current cell
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0.05, 1.5)
        new_x = cell.x + radius * np.cos(angle)
        new_y = cell.y + radius * np.sin(angle)
        new_cell = Cell(new_x, new_y, cell.vx, cell.vy)
        random_cell = np.random.choice(self.cells)   
        self.cells.remove(random_cell)
        self.cells.append(new_cell)


        
        
        
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
        if(np.random.uniform(0,1) < 0.5): 
            self.stickiness = CELL_STICKINESS_LOW
        else:
            self.stickiness = CELL_STICKINESS_HIGH
        
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
        self.cell_scatter = ax.scatter(self.cell_x, self.cell_y, c=self.cell_stickiness, cmap='inferno', s=50, edgecolor='white', vmin=0, vmax=CELL_STICKINESS_HIGH*1.2)
        if(DRAW_ARROW): self.cell_quiver = ax.quiver(self.cell_x, self.cell_y, self.cell_vx * 0.15, self.cell_vy * 0.15, angles='xy', scale_units='xy', scale=0.02, color='darkblue')
        plt.subplots_adjust(bottom=0.2)

        ax.set_xlim(0, WORLD_SIZE)
        ax.set_ylim(0, WORLD_SIZE)
        ax.set_aspect('equal', adjustable='box')
        ax.set_title(f"Timestep: 0")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        target_point=ax.scatter(sim.target_position[0], sim.target_position[1], c='orange', s=50, edgecolor='white')
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
            self.cell_quiver.set_UVC(self.cell_vx * 0.15, self.cell_vy * 0.15)        

    def redraw_plot(self, sim):
        self.update_cell_positions(sim)
        cell_scatter_new = self.ax.scatter(self.cell_x, self.cell_y, c=self.cell_stickiness, cmap='inferno', s=50, edgecolor='white', vmin=0, vmax=CELL_STICKINESS_HIGH*1.2)
        if(DRAW_ARROW): 
            cell_quiver_new = self.ax.quiver(self.cell_x, self.cell_y, self.cell_vx * 0.15, self.cell_vy * 0.15, angles='xy', scale_units='xy', scale=0.02, color='darkblue')
            self.cell_quiver.remove()
            self.cell_quiver = cell_quiver_new
        self.cell_scatter.remove()
        self.fig.canvas.draw_idle()
        self.cell_scatter = cell_scatter_new
        self.grid_im.remove()
        self.grid_im = self.ax.imshow(sim.grid.T, extent=(0, WORLD_SIZE, 0, WORLD_SIZE), origin='lower', cmap='viridis', alpha=1.0)
        
        plt.pause(0.01)
            
            
# 4. Execute the main loop
if __name__ == "__main__":
    # with cProfile.Profile() as pr:
        main()
        # pr.print_stats()