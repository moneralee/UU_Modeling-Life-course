import numpy as np
import matplotlib.pyplot as plt

# Define parameters
k = 2      # Initial value
j = 13     # Value to approach
r = 0.01   # Exponential rate of change

# Define the function
def smooth_function(t, k, j, r):
    return j + (k - j) * np.exp(-r*t)

# Generate time values
t_values = np.linspace(0, 1000, 500)

# Compute function values
f_values = smooth_function(t_values, k, j, r)

# Plot the function
plt.figure(figsize=(8, 6))
plt.plot(t_values, f_values, label=f'$f(t) = {j} + ({k} - {j}) \\cdot e^{{-r \\cdot t}}$')
plt.title('Smooth Function Visualization')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.axhline(y=j, color='r', linestyle='--', label='Asymptote j')
plt.legend()
plt.grid(True)
plt.show()