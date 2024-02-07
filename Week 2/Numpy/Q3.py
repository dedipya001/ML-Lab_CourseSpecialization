#Write a NumPy program to test whether each element of a 1-D array is also present in a second array.

import numpy as np
arr1 = np.array([1,3,5,6,3,21])
arr2 = np.array([1,2,3,4,5,6,7,8,9,0])

result = np.in1d(arr1, arr2)

print(result)