# Implement Principal Component Analysis Algorithm and use it to reduce dimensions of Iris Dataset (from 4D to 2D). Plot the scatter plot for samples in the transformed
# domain with different colour codes for samples belonging to different classes.

# Note: Develop the code without using library function ‘PCA’. Import linalg module from
# numpy to compute the eigen values and corresponding eigen vectors.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# PCA Implementation
def pca(X, n_components=2):
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov_matrix = np.cov(X_centered, rowvar=False)
    
    # Compute eigenvalues and eigenvectors
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_values = eigen_values[sorted_indices]
    sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
    
    # Select top n_components eigenvectors
    principal_components = sorted_eigen_vectors[:, :n_components]
    
    # Project the data onto the principal components
    transformed_data = np.dot(X_centered, principal_components)
    
    return transformed_data

# Perform PCA to reduce dimensionality to 2D
X_transformed = pca(X, n_components=2)

# Plot original and transformed data on the same plot
plt.figure(figsize=(10, 6))

# Original data plot
for i in range(len(iris.target_names)):
    plt.scatter(X[y == i, 0], X[y == i, 1], label='Original - ' + iris.target_names[i], alpha=0.5, marker='o')

# Transformed data plot
for i in range(len(iris.target_names)):
    plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], label='Transformed - ' + iris.target_names[i], alpha=0.5, marker='x')

plt.title('Original and Transformed Iris Dataset')
plt.xlabel('Feature 1 / Principal Component 1')
plt.ylabel('Feature 2 / Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
