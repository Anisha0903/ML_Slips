import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

dataset = pd.read_csv('Salary_positions.csv')

X = dataset[['Level']].values
y = dataset['Salary'].values

model = LinearRegression()
model.fit(X, y)

salary_level_11 = model.predict(np.array([[11]]))[0]
salary_level_12 = model.predict(np.array([[12]]))[0]

print(f"Predicted salary for level 11: {salary_level_11}")
print(f"Predicted salary for level 12: {salary_level_12}")
