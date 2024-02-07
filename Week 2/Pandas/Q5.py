# Write a Pandas program to calculate the number of characters in each word in a given series.


import pandas as pd
arr1 = pd.Series(['Red', 'Green', 'White', 'Black', 'Pink', 'Yellow'])
result = arr1.map(lambda x: len(x))
print(result)

