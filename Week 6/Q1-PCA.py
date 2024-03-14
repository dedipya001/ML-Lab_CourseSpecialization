# Consider the two dimensional data matrix [(2, 1), (3, 4), (5, 0), (7, 6), (9, 2)].Implement principal component analysis. Use this to obtain the feature in transformed 2D feature space. Plot the scatter plot of data points in both the original as well as transformed domain.

import numpy as np
from sklearn.decomposition import PCA


import matplotlib.pyplot as plt
data = np.array([(2, 1), (3, 4), (5, 0), (7, 6), (9, 2)])

# Perform PCA
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

# Plot original data
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.5)
plt.title('Original Data')
# plt.show()

# Plot transformed data
plt.figure(figsize=(6, 6))
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='red', alpha=0.5)
plt.title('Transformed Data')
# plt.show()

# Plot original and transformed data on the same plot
plt.figure(figsize=(6, 6))
plt.scatter(data[:, 0], data[:, 1], color='blue', alpha=0.5, label='Original Data')
plt.scatter(transformed_data[:, 0], transformed_data[:, 1], color='red', alpha=0.5, label='Transformed Data')
plt.title('Original and Transformed Data')
plt.legend()
plt.show()