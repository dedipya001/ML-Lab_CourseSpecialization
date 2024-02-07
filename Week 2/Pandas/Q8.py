# Write a Pandas program to select the rows where the number of attempts in the examination is greater than 2.

import pandas as pd
import numpy as np

data = {'Name': ['John', 'Emma', 'Mike', 'Sophia', 'Oliver'],
    'Attempts': [3, 2, 4, 1, 5]}

df = pd.DataFrame(data)

selected_rows = df[df['Attempts'] > 2]

print(selected_rows)




