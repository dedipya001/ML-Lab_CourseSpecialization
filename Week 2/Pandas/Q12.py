# Write a Pandas program to remove infinite values from a given DataFrame.

import pandas as pd
import numpy as np

data = pd.DataFrame([1000, 2000, 3000, -4000, np.inf, -np.inf])

data = data.replace([np.inf, -np.inf], np.nan)


data.columns = ['X']
print(data)


