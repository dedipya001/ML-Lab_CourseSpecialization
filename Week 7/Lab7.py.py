import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
# Load Iris dataset
iris_data = load_iris()
data = iris_data.data
target = iris_data.target
# Extract sepal length and sepal width columns
sepal_length = data[:, 0]
sepal_width = data[:, 1]
# Compute Covariance Matrix
covariance_matrix = np.cov(sepal_length, sepal_width)
# Compute Correlation Matrix
correlation_matrix = np.corrcoef(sepal_length, sepal_width)
print("Covariance Matrix:")
print(covariance_matrix)
print("\nCorrelation Matrix:")
print(correlation_matrix)
def custom_LDA(X, y, n_components=2):
    # Calculate the class means
    class_means = []
    for c in np.unique(y):
        class_means.append(np.mean(X[y == c], axis=0))
    # Compute within-class scatter matrix
    S_W = np.zeros((X.shape[1], X.shape[1]))
    for c, mean in zip(np.unique(y), class_means):
        class_scatter = np.zeros((X.shape[1], X.shape[1]))
        for row in X[y == c]:
            row, mean = row.reshape(-1, 1), mean.reshape(-1, 1)
            class_scatter += (row - mean).dot((row - mean).T)
        S_W += class_scatter
    # Compute between-class scatter matrix
    overall_mean = np.mean(X, axis=0)
    S_B = np.zeros((X.shape[1], X.shape[1]))
    for c, mean in zip(np.unique(y), class_means):
        n = X[y == c].shape[0]
        mean = mean.reshape(-1, 1)
        overall_mean = overall_mean.reshape(-1, 1)
        S_B += n * (mean - overall_mean).dot((mean - overall_mean).T)
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
    # Select the top k eigenvectors
    top_eigen_indices = np.argsort(eigenvalues)[::-1][:n_components]
    W = eigenvectors[:, top_eigen_indices]
    return W
# Applying custom LDA to reduce dimensions
X_lda = custom_LDA(data, target, n_components=2)
X_lda = data.dot(X_lda)
# Plotting the transformed data
plt.figure(figsize=(10, 6))
colors = ['blue', 'red', 'green']
for i, c in enumerate(np.unique(target)):
    plt.scatter(X_lda[target == c, 0], X_lda[target == c, 1], c=colors[i], label=iris_data.target_names[c])
plt.title('LDA Dimensionality Reduction of Iris Dataset')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend()
plt.show()
