import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("house_prices.csv")


df = df[["location", "Price (in rupees)"]]
df = df.dropna()

le = LabelEncoder()
df["location"] = le.fit_transform(df["location"].astype(str))

X = df[["location"]]
y = df["Price (in rupees)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MSE:", mse)

loc_value = le.transform(["mumbai"])
predicted_price = model.predict([[loc_value[0]]])
print("Predicted Price for Mumbai:", predicted_price[0])
