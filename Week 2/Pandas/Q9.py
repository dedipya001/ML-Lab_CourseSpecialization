#  Write a Pandas program to append a new row "k" to data frame with given values for each column. Now delete the new row and return the original DataFrame.

import numpy as np
import pandas as pd

data = {'Name': ['John', 'Emma', 'Mike', 'Sophia', 'Oliver'],
    'Attempts': [3, 2, 4, 1, 5]}
df = pd.DataFrame(data)


df.loc['k'] = ['Suresh', 15]
print(df)



df = df.drop('k')
print(df)


