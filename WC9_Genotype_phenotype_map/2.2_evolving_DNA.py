import random
import math
import matplotlib.pyplot as plt
from collections import Counter

# set the random number seed
random.seed(0)

plt.ion()  # Enable interactive mode

# Parameters
alphabet = "ATCG"
target_sequence = "GATGCGCGCTGGATTAAC"  # Example target sequence (seems random, no?)
dna_length = len(target_sequence)
target_length = len(target_sequence)

# Simulation settings
population_size = 500  # must be a square number for grid mode
generations = 20000
mutation_rate = 0.00005  # Probability of mutation per reproduction event
sample_interval = 5
sample_size = population_size
no_reproduction_chance = 1

# Core functions
def fitness(dna):
    return 1 - sum(a != b for a, b in zip(dna, target_sequence)) / target_length

def mutate(dna, rate=mutation_rate):
    return ''.join(
        random.choice([b for b in alphabet if b != base]) if random.random() < rate else base
        for base in dna
    )

def count_beneficial_mutations(dna):
    f0 = fitness(dna)
    count = 0
    for i in range(len(dna)):
        for b in alphabet:
            if b != dna[i]:
                mutant = dna[:i] + b + dna[i+1:]
                if fitness(mutant) > f0:
                    count += 1
    return count

def diversity(pop):
    counts = {}
    for ind in pop:
        counts[ind] = counts.get(ind, 0) + 1
    total = len(pop)
    return -sum((c/total) * math.log(c/total + 1e-9) for c in counts.values()) if total > 0 else 0

# Initialize population
initial_sequence = "GATAGCGAAGTTTAGCCG" # far from target (only first 3 are correct)
population = [initial_sequence for _ in range(population_size)]

avg_fitness = []
avg_beneficial = []
diversity_over_time = []
best_individuals = []

def get_neighbors(i, j):
    return [(x % side, y % side)
            for x in range(i-1, i+2)
            for y in range(j-1, j+2)]

# Initialize interactive plot
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel("Generation")
ax1.set_ylabel("Average Fitness", color='tab:blue')
ax1.set_ylim(0, 1)
line1, = ax1.plot([], [], color='tab:blue', linewidth=2, label='Fitness')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Beneficial Mutations / Diversity", color='tab:purple')
line2, = ax2.plot([], [], color='tab:purple', linestyle='--', linewidth=2, label='Beneficial Mutations')
line3, = ax2.plot([], [], color='tab:green', linestyle=':', linewidth=2, label='Diversity')
ax2.tick_params(axis='y', labelcolor='tab:purple')
fig.suptitle("Evolution Toward Target Sequence")
fig.tight_layout()
ax2.set_ylim(0, 20)
fig.legend(loc='upper right')
plt.grid(True)
plt.draw()

best_seq = ""
best_score = -1
found = False

# Evolution loop
for gen in range(generations):
    fitnesses = [fitness(ind) for ind in population]
    total_fit = sum(fitnesses)
    best = max(fitnesses)
    if(best == 1 and not found):
        found = True
        print("Found perfect solution at generation", gen)
        
    if gen % sample_interval == 0:
        sample = random.sample(population, sample_size)
        avg_beneficial.append(sum(count_beneficial_mutations(ind) for ind in sample) / sample_size)
        diversity_over_time.append(diversity(population))

        # Update plot data
        line1.set_data(range(len(avg_fitness)+1), avg_fitness + [sum(fitnesses)/population_size])
        line2.set_data(range(len(avg_beneficial)), avg_beneficial)
        line3.set_data(range(len(diversity_over_time)), diversity_over_time)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        best = max(population, key=fitness)
        fig.suptitle(f"Best: {best} (target: {target_sequence})", fontsize=14)
        plt.pause(0.01)

    else:
        avg_beneficial.append(avg_beneficial[-1])
        diversity_over_time.append(diversity_over_time[-1])

    # Roulette wheel selection (as in evolving_fitness_final.py)
    tournament_size = 10  # can be adjusted

    for _ in range(population_size):
        # Select tournament_size individuals randomly
        competitors = random.sample(population, tournament_size)
        # Pick the one with highest fitness
        fitness_values = [fitness(ind) for ind in competitors]
        total = sum(fitness_values)
        # Add a "no reproduction" dummy competitor with fitness = 0
        competitors_with_dummy = competitors + [None]
        probs = [f / total for f in fitness_values] + [no_reproduction_chance / total]
        winner = random.choices(competitors_with_dummy, weights=probs, k=1)[0]
        if winner is not None:
            # Mutate winner to produce offspring
            offspring = mutate(winner)
            # Remove a random individual from the population
            dead_idx = random.randrange(len(population))
            population[dead_idx] = offspring
        


    avg_fitness.append(sum(fitness(ind) for ind in population) / population_size)
    if gen % 250 == 0:
        best = max(population, key=fitness)
        best_individuals.append((gen, best))

input("\nSimulation complete. Press Enter to exit plot window...")