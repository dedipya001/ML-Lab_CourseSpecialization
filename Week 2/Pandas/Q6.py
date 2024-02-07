#Write a Pandas program to convert year-month string to dates adding a specified day of the month.

import pandas as pd
import numpy as np
from datetime import datetime

s = pd.Series(['Jan 2015', 'Feb 2016', 'Mar 2017', 'Apr 2018', 'May 2019'])
print("Original Series:")
print(s)

result = s.map(lambda x: datetime.strptime('01 ' + x, '%d %b %Y'))
print("\nNew dates:")
print(result)




