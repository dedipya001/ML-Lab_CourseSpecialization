# Given Dataset : traincsv.csv

#Apply the dimensionality reduction techniques such as missing values ratio, low variance,filter, and high correlation filter on the given data set.
#Note: For Missing Values Ratio (&gt;20%), For Low Variance and High Correlation reduce the dimension (12 to 8).



import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer

# Load the dataset
data = pd.read_csv("traincsv.csv")

# Identify categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

# One-hot encode categorical columns
data = pd.get_dummies(data, columns=categorical_columns)

# Missing Values Ratio
missing_values_ratio_threshold = 0.2
missing_values_percentage = data.isnull().mean()
columns_to_keep = missing_values_percentage[missing_values_percentage <= missing_values_ratio_threshold].index
data = data[columns_to_keep]

# Impute missing values
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Low Variance Filter
variance_threshold = 0.01
selector = VarianceThreshold(threshold=variance_threshold)
selector.fit(data)
columns_to_keep = data.columns[selector.get_support()]
data = data[columns_to_keep]

# High Correlation Filter
correlation_threshold = 0.8
correlation_matrix = data.corr().abs()
upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))
high_correlation_columns = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
data = data.drop(high_correlation_columns, axis=1)

# Further dimensionality reduction to achieve 8 features
if len(data.columns) > 8:
    k_best_selector = SelectKBest(score_func=f_regression, k=8)
    X = data.drop("Item_Outlet_Sales", axis=1)  # Replace "Item_Outlet_Sales" with the name of your target column
    y = data["Item_Outlet_Sales"]  # Replace "Item_Outlet_Sales" with the name of your target column
    k_best_selector.fit(X, y)
    columns_to_keep = X.columns[k_best_selector.get_support()]
    data = data[columns_to_keep]

# Now 'data' contains the reduced dimensional dataset
print("Dimensions after reduction:", data.shape)
print(data.head())
