# Write a Pandas program to convert Series of lists to one Series.

import pandas as pd
s = pd.Series([ 
    ['Red', 'Green', 'White'],
    ['Red', 'Black'],
    ['Yellow']])

s = s.apply(pd.Series).stack().reset_index(drop=True)
print(s)
