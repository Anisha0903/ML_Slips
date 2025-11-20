import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("house_price.csv")

X = data[['Area']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

area = float(input("Enter house area in sqft: "))
pred = model.predict([[area]])
print("Predicted House Price:", pred[0])
