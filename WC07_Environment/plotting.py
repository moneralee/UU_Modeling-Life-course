import numpy as np
import matplotlib.pyplot as plt

# Adjusted Parameters
pa = 1.0
da = 1.0
Ka = 0.35  # Adjusted to change the shape of func_A
pb = 1.0
db = 1.0
Kb = 0.35  # Adjusted to change the shape of func_B

# Function definitions
def func_A(B, pa, da, Ka):
    return (pa / da) * Ka**2 / (Ka**2 + B**2)

def func_B(A, pb, db, Kb):
    return (pb / db) * Kb**2 / (Kb**2 + A**2)

# Generate values for A and B
A_values = np.linspace(0, 2, 500)
B_values = np.linspace(0, 2, 500)

# Calculate corresponding B for A and A for B
B_from_A = func_B(A_values, pb, db, Kb)
A_from_B = func_A(B_values, pa, da, Ka)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(A_values, B_from_A, 'b-', label='B null cline (Blue Line)')
plt.plot(A_from_B, B_values, 'r-', label='A null cline (Red Line)')
plt.xlabel('A')
plt.ylabel('B')
plt.title('Plot of A and B')
plt.legend()

# Set axis limits
plt.xlim(0, 1.5)  # Start x-axis at 0
plt.ylim(0, 1.5)  # Start y-axis at 0

plt.show()