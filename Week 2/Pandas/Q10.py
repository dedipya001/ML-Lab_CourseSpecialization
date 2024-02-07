# Write a Pandas program to sort the DataFrame first by name in descending order, then by score in ascending order.

import numpy as np
import pandas as pd 

data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
                'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
                'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
                'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

df = pd.DataFrame(data, index=labels)
print(df)
print(" ")
df = df.sort_values(by=['name'], ascending=[False])
print(df)
print(" ")
df = df.sort_values(by=['score'], ascending=[True])
print(df)




