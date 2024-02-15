# Implement Linear Regression and calculate sum of residual error on the following Datasets.
# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# y = [1, 3, 2, 5, 7, 8, 8, 9, 10, 12]

# 1. Compute the regression coefficients using analytic formulation and calculate Sum
# Squared Error (SSE) and R^2 value.

# 2.  Implement gradient descent (both Full-batch and Stochastic with stopping
# criteria) on Least Mean Square loss formulation to compute the coefficients of
# regression matrix and compare the results using performance measures such as R 2
# SSE etc.

import numpy as np

x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

X = np.column_stack((np.ones(len(x)), x))

coefficients = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

y_pred = X.dot(coefficients)

sse = np.sum((y - y_pred) ** 2)      # Sum Squared Error (SSE)


y_mean = np.mean(y)
sst = np.sum((y - y_mean) ** 2)  # R^2 value
r_squared = 1 - (sse / sst)

print("Regression Coefficients:", coefficients)
print("Sum Squared Error (SSE):", sse)
print("R^2 value:", r_squared)
