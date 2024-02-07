# Write a Pandas program to find the positions of numbers that are multiples of 5 of a given series.
import pandas as pd
import numpy as np
arr1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

x = np.where(arr1 % 5 == 0)[0]

print(x)
result = ', '.join(map(str, x))
print(result)