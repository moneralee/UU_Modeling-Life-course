import random
import math
import matplotlib.pyplot as plt

# Set the random number seed for reproducibility
random.seed(0)

plt.ion()  # Enable interactive plotting

# --- PARAMETERS ---
initial_fitness = 0.1            # Starting fitness for all individuals
population_size = 500             # Number of individuals (should be a square number for grid mode)
generations = 20000               # Number of generations to simulate
mutation_rate = 0.005            # Probability of mutation per reproduction event
sample_interval = 5               # How often to sample and plot data

# --- INITIALIZATION ---
# Create initial population: all individuals start with the same fitness
population = [initial_fitness for _ in range(population_size)]

# Lists to track average fitness and diversity over time
avg_fitness = []
diversity_over_time = []

# --- CORE FUNCTIONS ---

def mutate(fitness, rate=mutation_rate):
    """Mutate the fitness value with a given probability."""
    if random.random() < rate:
        # Fitness changes by a random value in [-0.1, 0.1], clipped to [0, 1]
        return min(1.0, max(0.0, fitness + random.uniform(-0.1, 0.1)))
    return fitness

def calculate_diversity(population):
    """NOT YET IMPLEMENTED! Calculate diversity as the standard deviation of fitness values."""
    return 0 

# --- PLOTTING SETUP ---
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel("Generation")
ax1.set_ylabel("Average Fitness", color='tab:blue')
ax1.set_ylim(0, 1)
line1, = ax1.plot([], [], color='tab:blue', linewidth=2, label='Fitness')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Second y-axis for diversity
ax2 = ax1.twinx()
ax2.set_ylabel("Diversity", color='tab:green')
line2, = ax2.plot([], [], color='tab:green', linestyle=':', linewidth=2, label='Diversity')
ax2.tick_params(axis='y', labelcolor='tab:green')

fig.suptitle("Evolution Toward Fitness 1")
fig.tight_layout()
fig.legend(loc='upper right')
plt.grid(True)
plt.draw()

# --- EVOLUTION LOOP ---
best_fitness = -1
found = False

for gen in range(generations):
    total_fit = sum(population)
    best = max(population)
    # Print when a perfect solution is found
    if best == 1 and not found:
        found = True
        print("Found perfect solution at generation", gen)
        
    # Sample and plot data at intervals
    if gen % sample_interval == 0:
        avg_fitness.append(total_fit / population_size)
        diversity_over_time.append(calculate_diversity(population))
        x_vals = [i * sample_interval for i in range(len(avg_fitness))]
        line1.set_data(x_vals, avg_fitness)
        line2.set_data(x_vals, diversity_over_time)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        fig.suptitle(f"Best Fitness: {best:.2f}", fontsize=14)
        plt.pause(0.01)
        

    # --- MORAN PROCESS ---
    # For each individual, perform a reproduction event
    for _ in range(100):  # 100 competition events per generation
        # Select 1 random individual for replication
        probs = [fit for fit in population] # All probability weights are equal (1.0)
        parent_idx = random.choices(range(len(population)), weights=probs)[0] # Grab one random individual based on an unweighted list...
        # Select individual to be replaced (uniform random)
        dead_idx = random.randrange(len(population))
        # Copy population for next generation
        new_pop = population.copy()
        # Offspring replaces the dead individual (with possible mutation)
        new_pop[dead_idx] = mutate(population[parent_idx])
        population = new_pop

input("\nSimulation complete. Press Enter to exit plot window...")