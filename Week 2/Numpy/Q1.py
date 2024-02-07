# Write a NumPy program to multiply two given arrays of same size element-by-element.



# taking inputs in the starting
import numpy as np

arr1 = np.array([1, 2, 3, 4])
arr2 = np.array([5, 6, 7, 8])

result = arr1 * arr2

print(result)

#taking input from the user

arr3 = np.array([ x for x in input("Enter the elements of the first array: ").split()])
arr4 = np.array([ x for x in input("Enter the elements of the second array: ").split()])

result2 = np.multiply(arr3 ,arr4)

print(result2)


    




