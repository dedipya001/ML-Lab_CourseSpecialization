# Build a Soft Margin Linear SVM classifier and evaluate the model.
# Note: You can generate the synthetic dataset using make_blobs from sklearn.datasets.

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

X, y = make_blobs(n_samples=1000, centers=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train, y_train)


y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(" ")
print("Accuracy:", accuracy)


print(" ")


report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
