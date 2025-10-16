from rolling_clock import Clock # import the Clock class from rolling_clock.py
from graphics import * # import the visualization functions from graphics.py

# Similation parameters
totaltime=50*3600
divisiontime=1 # fraction of simulation time divisions are allowed
dt=1
time_steps = int(totaltime / dt)
time_visualization = 600 # How often to visualize the tissue growth in seconds

growth_rate =128/(10*3600)
cell_width=4
doubling_threshold = 2*cell_width

#when modifying the degradation rate, modify the production rate
#similarly, that way p/d stays constant and you have same max level
#at the right end of the tissue if no growth takes place
pfgf=60*0.001
dfgf=4*0.00001

# Parameters for the clock
alpha = 1.0/60
beta = 1.0/60
K = 1.0
n=4
tau=20*60
mu=0.03/60
v=0.03/60


class Cell:
    def __init__(self, fgf=0, m=0, p=0):
        self.fgf = fgf
        self.xmin = 0
        self.xmax = 0
        self.xheight = 0
        self.clock = Clock(alpha, beta, K, n, tau, mu, v, m0=m, p0=p)  # Initialize the clock for each cell
    
    def grow_cell(self, dt, growth_rate):
        self.fgf = self.fgf * self.xheight / (self.xheight + dt * growth_rate)  # Dilute content with volume
        self.xmax += dt * growth_rate
        self.xheight = self.xmax - self.xmin

    def update_clock_tau(self):
        self.clock.set_tau((1+0.5*(100-self.fgf)/100)*tau) # Update tau based on FGF level and global tau

class Tissue:
    def __init__(self, doubling_threshold=doubling_threshold, time_visualization=time_visualization, growth_rate=growth_rate, dt=dt):
        self.cells = []
        self.num_cells = 0
        self.xmax_tissue = 0
        self.doubling_threshold = doubling_threshold
        self.time_visualization = time_visualization
        self.growth_rate = growth_rate
        # self.dt=dt

    # some functions to update the tissue properties
    def update_xmax_tissue(self):
        if self.cells:
            self.xmax_tissue = max(cell.xmax for cell in self.cells)
        
    # Functions to add, insert, and divide cells
    def add_cell(self, cell):
        self.cells.append(cell)
        self.num_cells += 1
        self.update_xmax_tissue()
    
    def insert_cell(self, cell, index):
        # Insert a cell at the specified index
        self.cells.insert(index, cell)
        self.num_cells += 1
        self.update_xmax_tissue()

    def divide_cell(self, cell_index):
        # Divide the cell at the specified index
        cell = self.cells[cell_index]
        mid_point = (cell.xmin + cell.xmax) / 2
        new_cell = Cell(fgf=cell.fgf)  # Create a new cell with the same FGF
        new_cell.clock=cell.clock.__copy__()  # Copy the clock state to the new cell

        new_cell.xmin = mid_point
        new_cell.xmax = cell.xmax
        new_cell.xheight = new_cell.xmax - new_cell.xmin
        # Adjust the original cell
        cell.xmax = mid_point
        cell.xheight = cell.xmax - cell.xmin
        # Add the new cell to the list of cells
        self.insert_cell(new_cell, cell_index + 1)  # Insert after the original cell
        return new_cell


    # functions to initialize and grow the tissue 
    def initialize_regular_tissue(self, num_cells=1, initial_fgf=100, cell_width=cell_width):
        # Initialize a regular tissue with a specified number of cells of which the first has initial FGF
        # and the rest have FGF=0
        for i in range(num_cells):
            if i <1:
                cell = Cell(fgf=initial_fgf)  # First cell has initial FGF
            else:
                cell = Cell(fgf=0)  # Rest have FGF=0
            cell.xmin = i * cell_width
            cell.xmax = (i + 1) * cell_width
            cell.xheight = cell_width
            self.add_cell(cell)
        self.update_xmax_tissue()

    def grow_tissue(self, dt, growth_rate, doubling_threshold,n=1):
        for cell_index,cell in enumerate(self.cells[-n:],-n):  # Apply growth to the n last cells only
            # Grow the cell
            cell.grow_cell(dt, growth_rate)
            # Update xmin and xmax of all other cells above it "pushing" them
            if cell_index != - 1:  # Avoid index out of range
                for other_cell in self.cells[cell_index+1:]:
                    if other_cell.xmin > cell.xmin and other_cell.xmin < cell.xmax:
                        other_cell.xmin += dt*growth_rate
                        other_cell.xmax += dt*growth_rate
                    elif other_cell.xmin >= cell.xmax:
                        other_cell.xmin += dt*growth_rate
                        other_cell.xmax += dt*growth_rate
            self.update_xmax_tissue()


        # After growing the tissue, check for cell division
        for cell_index, cell in enumerate(self.cells):
            # Check if the cell has reached the doubling threshold
            if cell.xheight >= doubling_threshold:
                # Print which cell is being divided
                # print(f"Dividing cell at index: {cell_index}")
                # Create a new cell for division
                self.divide_cell(cell_index)
                # Add the new cell to the list of new cells

            # print(f"Current number of cells: {self.num_cells}")


    # Simulate the morphogen gradient over time
    # The last cell produces fgf, the others only degrade it
    def simulate_morphogen(self, time_steps, pfgf, dfgf):
        for i, cell in enumerate(self.cells):
            if i == len(self.cells) - 1:  # only last cell produces fgf
                dtfgf = dt * (pfgf  - (dfgf * cell.fgf))
            else: # other cells degrade fgf
                dtfgf = dt * (- (dfgf * cell.fgf))
            cell.fgf += dtfgf

    def run_clocks(self, t, dt):
        # Update the clocks of all cells
        for cell in self.cells:
            cell.update_clock_tau()
            cell.clock.simulate(t, dt)

# Modify simulate_development to include plotting
def simulate_development(time_steps):
    # Initialize a tissue
    tissue = Tissue()
    # Initialize the tissue with a single cell
    tissue.initialize_regular_tissue(num_cells=1, initial_fgf=100, cell_width=cell_width)

    plt.ion()
    tissueplot=TissuePlot(tissue, time_visualization, dt,  n_axis=2)
    tissueplot.initialize_axis_cell_data(axis_index=0, colormap='viridis', attribute='fgf',update_direction={"max_up":1,"max_down":0.75 }, label='FGF Level')
    tissueplot.initialize_axis_cell_data(axis_index=1, colormap='Greys', attribute='clock.p_values[-1]',update_direction={"max_up":1.3,"max_down":0.6} , label='P Level')

    for t in range(time_steps):
        if t / time_steps < divisiontime:  # Allow divisions only in the first fraction of the simulation time
            tissue.grow_tissue(dt, growth_rate, doubling_threshold)
        tissue.simulate_morphogen(time_steps, pfgf, dfgf)   
        tissue.run_clocks(t * dt, dt)     
        if t *dt % time_visualization == 0:  # Plot every something timesteps
            tissueplot.update_xmax_display() # Update the maximum x-axis limit for display purposes
            tissueplot.update_plot_cell_data(t,axis_index=0)
            tissueplot.update_plot_cell_data(t,axis_index=1)

    print("Simulation finished")
    # Finalize and show the plot
    plt.ioff()
    # plt.show()

# Call the function to simulate tissue growth and plot
simulate_development(time_steps)


