import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\91983\\OneDrive\\Desktop\\Sem 6\\Machine Learning Lab\\Week 1\\iris.csv", names=[
                 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class'])
print(df.head())

print(" ")

# --Sepal Length--
print("Sepal Length Stats: ")

mean_sepal_length = df['sepal_length'].mean()
print("Mean of sepal length: ", mean_sepal_length)

max_sepal_length = df['sepal_length'].max()
print("Max of sepal length: ", max_sepal_length)

min_sepal_length = df['sepal_length'].min()
print("Min of sepal length: ", min_sepal_length)

sd_sepal_length = df['sepal_length'].std()
print("Standard Deviation of sepal length: ", sd_sepal_length)
print(" ")

# --Petal Length--
print(" ")
print("Petal Length Stats: ")
mean_petal_length = df['petal_length'].mean()
print("Mean of petal length: ", mean_petal_length)

max_petal_length = df['petal_length'].max()
print("Max of petal length: ", max_petal_length)

min_petal_length = df['petal_length'].min()
print("Min of petal length: ", min_petal_length)

sd_petal_length = df['petal_length'].std()
print("Standard Deviation of petal length: ", sd_petal_length)

print(" ")

# --Sepal Width-
print(" ")
print("Sepal Width Stats: ")

mean_sepal_width = df['sepal_width'].mean()
print("Mean of sepal width: ", mean_sepal_width)

max_sepal_width = df['sepal_width'].max()
print("Max of sepal width: ", max_sepal_width)

min_sepal_width = df['sepal_width'].min()
print("Min of sepal width: ", min_sepal_width)

sd_sepal_width = df['sepal_width'].std()
print("Standard Deviation of sepal width: ", sd_sepal_width)

print(" ")


# --Petal Width--

print(" ")
print("Petal Width Stats: ")
mean_petal_width = df['petal_width'].mean()
print("Mean of petal width: ", mean_petal_width)

max_petal_width = df['petal_width'].max()
print("Max of petal width: ", max_petal_width)

min_petal_width = df['petal_width'].min()
print("Min of petal width: ", min_petal_width)

sd_petal_width = df['petal_width'].std()
print("Standard Deviation of petal width: ", sd_petal_width)

print(" ")


setosa_df = df[df['class'].isin(['Iris-setosa','Iris-versicolor'])]
versicolour_df = df[df['class'].isin(['Iris-versicolor',"Iris-virginica"])]
virginica_df = df[df['class'].isin(['Iris-virginica',"Iris-setosa"])]

setosa_versicolour_corr = setosa_df['sepal_length'].corr(setosa_df["petal_length"])
versicolour_virginica_corr = versicolour_df[['sepal_length', 'petal_length']].corr().iloc[0, 1]
virginica_setosa_corr = virginica_df[['sepal_length', 'petal_length']].corr().iloc[0, 1]

# Display the results
print(f"Class correlation (Iris Setosa - Iris Versicolour): {setosa_versicolour_corr}")
print(f"Class correlation (Iris Versicolour - Iris Virginica): {versicolour_virginica_corr}")
print(f"Class correlation (Iris Virginica - Iris Setosa): {virginica_setosa_corr}")