# Write a Pandas program to create a subset of a given series based on value and condition.

import pandas as pd

data = pd.Series([10, 20, 30, 40, 50])

subset = data[data > 30]

print(subset)
