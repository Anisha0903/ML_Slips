import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import r2_score

data = pd.read_csv("boston_houses.csv")

X = data[['RM']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_r2 = r2_score(y_test, ridge_pred)
ridge_price_5 = ridge.predict([[5]])[0]

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_price_5 = lasso.predict([[5]])[0]

print("Ridge R2:", ridge_r2)
print("Lasso R2:", lasso_r2)
print("Ridge predicted price for 5 rooms:", ridge_price_5)
print("Lasso predicted price for 5 rooms:", lasso_price_5)
