import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

df = pd.read_csv("Salary_positions.csv")

X = df[["Level"]]
y = df["Salary"]

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)
linear_r2 = r2_score(y, y_pred_linear)

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)
poly_r2 = r2_score(y, y_pred_poly)

print("Simple Linear Regression R2:", linear_r2)
print("Polynomial Linear Regression R2:", poly_r2)

levels = np.array([[11], [12]])
levels_poly = poly.transform(levels)

print("Simple Linear Prediction Level 11:", lin_reg.predict([[11]])[0])
print("Simple Linear Prediction Level 12:", lin_reg.predict([[12]])[0])

print("Polynomial Prediction Level 11:", poly_reg.predict(levels_poly)[0])
print("Polynomial Prediction Level 12:", poly_reg.predict(levels_poly)[1])
