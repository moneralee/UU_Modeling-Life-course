import random
import math
import matplotlib.pyplot as plt
from collections import Counter

# set the random number seed
random.seed(0)

plt.ion()  # Enable interactive mode

# Codon table
codon_table = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L', 'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S', 'AGT': 'S', 'AGC': 'S',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'TAT': 'Y', 'TAC': 'Y', 'CAT': 'H', 'CAC': 'H',
    'CAA': 'Q', 'CAG': 'Q', 'AAT': 'N', 'AAC': 'N',
    'AAA': 'K', 'AAG': 'K', 'GAT': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E', 'TGT': 'C', 'TGC': 'C',
    'TGG': 'W', 'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
    'TAA': '*', 'TAG': '*', 'TGA': '*'
}

# Parameters
alphabet = "ATCG"
target_protein = "DARWIN"
dna_length = len(target_protein)*3
target_length = len(target_protein)

# Simulation settings
population_size = 500  # must be a square number for grid mode
generations = 20000
mutation_rate = 0.001   
sample_interval = 1
sample_size = 50
no_reproduction_chance = 0.01

# Core functions
def translate(dna):
    return ''.join(codon_table.get(dna[i:i+3], '?') for i in range(0, len(dna), 3))

def fitness(dna):
    protein = translate(dna)
    return 1 - sum(a != b for a, b in zip(protein, target_protein)) / target_length

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
    counts = Counter(pop)
    total = len(pop)
    return -sum((c/total) * math.log(c/total + 1e-9) for c in counts.values()) if total > 0 else 0

# Initialize population
initial_sequence = ''.join(random.choice(alphabet) for _ in range(dna_length))
population = [initial_sequence for _ in range(population_size)]

avg_fitness = []
avg_beneficial = []
diversity_over_time = []
best_individuals = []



# Initialize interactive plot
fig, ax1 = plt.subplots(figsize=(12, 8))
ax1.set_xlabel("Generation")
ax1.set_ylabel("Average Fitness", color='tab:blue')
ax1.set_ylim(0, 1)
line0, = ax1.plot([], [], color='black', linewidth=1, label='Max fitness')
line1, = ax1.plot([], [], color='tab:blue', linewidth=2, label='Fitness')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel("Beneficial Mutations / Diversity", color='tab:purple')
line2, = ax2.plot([], [], color='tab:purple', linestyle='--', linewidth=2, label='Beneficial Mutations')
line3, = ax2.plot([], [], color='tab:green', linestyle=':', linewidth=2, label='Diversity')
ax2.tick_params(axis='y', labelcolor='tab:purple')
fig.suptitle("Evolution Toward " + str(target_protein))
fig.tight_layout()
ax2.set_ylim(0, 5)
fig.legend(loc='upper right')
plt.grid(True)
plt.draw()

best_seq = ""
best_score = -1
best_fitnesses = []
found = False

# Evolution loop
for gen in range(generations):
    fitnesses = [fitness(ind) for ind in population]
    total_fit = sum(fitnesses)
    best = max(fitnesses)
    best_fitnesses.append(best)
    if(best == 1 and not found):
        found = True
        print("Found perfect solution at generation", gen)
        
    if gen % sample_interval == 0:
        sample = random.sample(population, sample_size)
        avg_beneficial.append(sum(count_beneficial_mutations(ind) for ind in sample) / sample_size)
        diversity_over_time.append(diversity(population))

        # Update plot data
        line0.set_data(range(len(avg_fitness)+1), best_fitnesses)
        line1.set_data(range(len(avg_fitness)+1), avg_fitness + [sum(fitnesses)/population_size])
        line2.set_data(range(len(avg_beneficial)), avg_beneficial)
        line3.set_data(range(len(diversity_over_time)), diversity_over_time)
        ax1.relim(); ax1.autoscale_view()
        ax2.relim(); ax2.autoscale_view()
        best = max(population, key=fitness)
        fig.suptitle(f"Best: {translate(best)} (target: {target_protein})", fontsize=14)
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
        if total == 0:
            continue
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
        best_individuals.append((gen, translate(best)))

input("\nSimulation complete. Press Enter to exit plot window...")