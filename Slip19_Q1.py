import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

data = pd.read_csv("Salary_positions.csv")

X = data[['Level']]
y = data['Salary']

slr = LinearRegression()
slr.fit(X, y)
slr_pred = slr.predict(X)
slr_r2 = r2_score(y, slr_pred)

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
plr = LinearRegression()
plr.fit(X_poly, y)
plr_pred = plr.predict(X_poly)
plr_r2 = r2_score(y, plr_pred)

print("Simple Linear Regression R2:", slr_r2)
print("Polynomial Regression R2:", plr_r2)

print("SLR Prediction Level 11:", slr.predict([[11]])[0])
print("SLR Prediction Level 12:", slr.predict([[12]])[0])

print("Polynomial Prediction Level 11:", plr.predict(poly.transform([[11]]))[0])
print("Polynomial Prediction Level 12:", plr.predict(poly.transform([[12]]))[0])
