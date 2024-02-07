# Write a NumPy program to save a NumPy array to a text file.
import numpy as np
arr1 = np.array([[1, 2, 3, 4],[5,6,7,8]])

np.savetxt('array.txt', arr1, delimiter = ',')
print("The array has been saved to the file 'array.txt'")