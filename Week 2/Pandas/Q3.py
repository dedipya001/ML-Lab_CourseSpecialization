# Write a Pandas program to display most frequent value in a given arr1 and replace everything else as "Other" in the arr1.


import pandas as pd


arr1 = pd.arr1([1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 8, 9, 9, 9])
most_frequent = arr1.mode()[0]
arr1 = arr1.apply(lambda x: x if x == most_frequent else "Other")
print(arr1)