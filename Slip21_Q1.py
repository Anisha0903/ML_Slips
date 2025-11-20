import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("house_price.csv")

X = data.drop('Price', axis=1)
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Predicted Prices:\n", predictions)

area = float(input("Enter area: "))
bed = int(input("Enter bedrooms: "))
age = float(input("Enter house age: "))

new_pred = model.predict([[area, bed, age]])
print("Predicted House Price:", new_pred[0])
