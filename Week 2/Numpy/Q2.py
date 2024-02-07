# Write a NumPy program to find the missing data in a given array.
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, np.nan, 8])

result = np.isnan(arr2)
result_ind = np.where(result == True)

print(f"The missing data is at index {result_ind} in the array {arr2}")
