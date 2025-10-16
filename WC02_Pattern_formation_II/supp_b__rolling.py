import numpy as np

# 1D Array Example
array_1d = np.array([1, 2, 3, 4, 5])
print("Original 1D Array:")
print(array_1d)

# Roll the 1D array by 2 positions
rolled_1d = np.roll(array_1d, 2)
print("\nRolled 1D Array by 2 positions:")
print(rolled_1d)

# 2D Matrix Example
matrix_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("\nOriginal 2D Matrix:")
print(matrix_2d)

# Roll the 2D matrix by 1 position along axis 0 (vertically)
rolled_2d_axis0 = np.roll(matrix_2d, 1, axis=0)
print("\nRolled 2D Matrix by 1 position along axis 0:")
print(rolled_2d_axis0)

# Roll the 2D matrix by 1 position along axis 1 (horizontally)
rolled_2d_axis1 = np.roll(matrix_2d, 1, axis=1)
print("\nRolled 2D Matrix by 1 position along axis 1:")
print(rolled_2d_axis1)