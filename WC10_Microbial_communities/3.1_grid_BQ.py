###
# PRACTICAL 1.2 SIMULATING IN DISCRETE SPACE
###

## Structure of code:
# 1. Import libraries
# 2. Set parameters
# 3. Initialize grid
# 4. Define simulation functions
# 5. Set up interactive plot (and function)
# 6. Run simulation (up until T=timesteps)
# 7. Show final plot

# 1. Load the required libraries
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.widgets import Slider, Button, RadioButtons

# 2. Parameters for simulation
grid_size = 50  # Resolution of the grid
bitstring_length = 10  # Length of the bitstring for each individual
death_prob = 0.1  # Probability of death
timesteps = 25000  # Number of timesteps to simulate
mixing = False  # Mixing individuals after every timestep
random.seed(56)  # Sets the random-number generator for reproducability 
c = 0.01  # Cost per 1-bit
nothing_happens = 2
mutation_rate_loss = 0.01   # 1 -> 0 (loss) is easy
mutation_rate_gain = 0.001  # 0 -> 1 (gain) is hard
                
# 3. Initialize a grid 
# Each cell is either None (empty) or a numpy array of bits (0/1)
grid = np.empty((grid_size, grid_size), dtype=object)
# Start with one 'living' in the middle, all bits set to 1
# Fill a small circle in the center with individuals (all 1s)
center = grid_size // 2
radius = grid_size // 10  # Adjust as needed for "small circle"
for i in range(grid_size):
    for j in range(grid_size):
        if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
            grid[i, j] = np.ones(bitstring_length, dtype=int)

# 4. Define functions

def can_replicate(bitarr, neighbors):
    """Check if this individual can replicate (public good model)."""
    if bitarr is None:
        return False
    # If all bits are 1, can always replicate
    if np.all(bitarr == 1):
        return True
    # Otherwise, check if for every 0 in bitarr, at least one neighbor has a 1 at that position
    for idx, val in enumerate(bitarr):
        if val == 0:
            if not any(n is not None and n[idx] == 1 for n in neighbors):
                return False
    return True



def calculate_fitness(bitarr):
    """Fitness is 1 - b * number of 1s. Empty cells (None) have fitness 0."""
    if bitarr is None:
        return 0.0
    return max(0.0, 1.0 - c * np.sum(bitarr))

def simulate_step(grid):
    new_grid = grid.copy()
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] is not None:  # Occupied site
                if random.random() < death_prob:
                    new_grid[i, j] = None
            else:  # Empty site
                neighbors_idx = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
                valid_idx = [(ni, nj) for ni, nj in neighbors_idx if 0 <= ni < grid_size and 0 <= nj < grid_size]
                neighbors = [grid[ni, nj] for ni, nj in valid_idx]
                # Only allow replication if at least one neighbor can_replicate (public good model)
                replicators = [n for n in neighbors if can_replicate(n, neighbors)]
                if not replicators:
                    continue  # No one can replicate, nothing happens
                # Compute fitness for each replicator
                fitnesses = [calculate_fitness(n) for n in replicators]
                total_fitness = sum(fitnesses) + nothing_happens
                pick = random.uniform(0, total_fitness)
                cumulative = 0.0
                chosen = None
                for idx, fit in enumerate(fitnesses):
                    cumulative += fit
                    if pick < cumulative:
                        chosen = replicators[idx]
                        break
                if chosen is None:
                    continue  # "Nothing happens" slice
                offspring = chosen.copy()
                
                for bidx in range(bitstring_length):
                    if offspring[bidx] == 1:
                        if random.random() < mutation_rate_loss:
                            offspring[bidx] = 0
                    else:
                        if random.random() < mutation_rate_gain:
                            offspring[bidx] = 1
                new_grid[i, j] = offspring
    return new_grid

def perfect_mix(grid):
    # Get all positions in the grid
    positions = [(i, j) for i in range(grid.shape[0]) for j in range(grid.shape[1])]
    # Extract all non-empty cells (individuals)
    individuals = [grid[i, j] for i, j in positions if grid[i, j] is not None]
    # Shuffle the positions
    random.shuffle(positions)
    # Create a new empty grid
    new_grid = np.empty_like(grid, dtype=object)
    # Assign each individual to a random position (first N positions)
    for ind, pos in zip(individuals, positions):
        new_grid[pos] = ind
    # All other positions remain None
    return new_grid

# 5. Set up interactive plot

plt.ion()
fig, (ax, ax_pop) = plt.subplots(1, 2, figsize=(14, 6))  # Wider figure for more spacing
plt.subplots_adjust(wspace=0.35)  # Increase spacing between plots

# Make death-slider half as wide, only under the grid
death_slider_ax  = fig.add_axes([0.15, 0.05, 0.32, 0.03], facecolor='blue') 
death_slider = Slider(death_slider_ax, 'death rate', 0.0, 1.0, valinit=death_prob) 
def set_death(val): 
    global death_prob
    death_prob = death_slider.val
death_slider.on_changed(set_death)

mixing_button_ax = fig.add_axes([0.05,0.1,0.1,0.05], facecolor='black')
mixing_button = Button(mixing_button_ax, 'mix')
def toggle_mix(val):
    global mixing
    mixing = not mixing
mixing_button.on_clicked(toggle_mix)

from matplotlib.colors import ListedColormap
# Custom colormap: gray for 0 ones, white for empty, then viridis for 1..bitstring_length
viridis_colors = plt.cm.viridis(np.linspace(0, 1, bitstring_length+1))
cmap = ListedColormap(['white', 'red'] + [viridis_colors[i] for i in range(1, bitstring_length+1)])

def plot_grid(grid, t, pop_sizes, genotype_counts):
    # Build an array for plotting: 0 for empty, 1..bitstring_length for number of 1s
    plot_arr = np.zeros((grid_size, grid_size), dtype=int)
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] is not None:
                plot_arr[i, j] = int(np.sum(grid[i, j])) + 1  # shift by 1: 0 ones -> 1, 1 ones -> 2, etc.
            # else: remains 0 (empty)
    ax.clear()
    im = ax.imshow(plot_arr, cmap=cmap, vmin=0, vmax=bitstring_length+1)
    ax.set_title(f"Timestep: {t}")
    ax.axis("off")
    # Add colorbar only once
    if not hasattr(plot_grid, "colorbar"):
        ticks = list(range(0, bitstring_length+2))
        ticklabels = ['empty', '0'] + [str(i) for i in range(1, bitstring_length+1)]
        plot_grid.colorbar = plt.colorbar(im, ax=ax, fraction=0.1, ticks=ticks)
        plot_grid.colorbar.set_label("Number of 1s in bitstring")
        plot_grid.colorbar.ax.set_yticklabels(ticklabels)
    # Plot population size
    ax_pop.clear()
    ax_pop.plot(pop_sizes, color='black', label="Total", linewidth=2)
    # Plot genotype counts for each number of 1s
    for k in range(bitstring_length+1):
        counts = [genotype_counts[tidx][k] for tidx in range(len(genotype_counts))]
        ax_pop.plot(counts, color=viridis_colors[k], label=f"{k}", alpha=0.7)
    ax_pop.set_title("Population size and genotype distribution")
    ax_pop.set_xlabel("Timestep")
    ax_pop.set_ylabel("Number of individuals")
    ax_pop.set_xlim(0, len(pop_sizes))
    ax_pop.set_ylim(0, grid_size * grid_size)
    ax_pop.legend(loc="upper right", fontsize="small", ncol=2)

# 6. Run simulation
pop_sizes = []
genotype_counts = []
for t in range(1, timesteps + 1):
    # Count number of non-empty cells
    pop_size = sum(1 for i in range(grid_size) for j in range(grid_size) if grid[i, j] is not None)
    pop_sizes.append(pop_size)
    # Count number of individuals with k ones for k=0..bitstring_length
    counts = [0] * (bitstring_length + 1)
    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] is not None:
                ones = int(np.sum(grid[i, j]))
                counts[ones] += 1
    genotype_counts.append(counts)
    if t%20 == 0:
        plot_grid(grid, t, pop_sizes, genotype_counts)
        plt.pause(0.01)
    grid = simulate_step(grid)
    if mixing:
        grid = perfect_mix(grid)

plt.ioff()
plt.show()