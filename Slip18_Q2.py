import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("salary_position.csv")

X = data[['Level']]
y = data['Salary']

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

level = int(input("Enter position level: "))
pred = model.predict(poly.transform([[level]]))
print("Predicted Salary:", pred[0])
