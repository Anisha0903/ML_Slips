import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso

df = pd.read_csv("boston-housing-dataset.csv")

X = df[['RM']]
y = df['AGE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

ridge_pred_5 = ridge.predict([[5]])[0]
lasso_pred_5 = lasso.predict([[5]])[0]

print("Ridge Prediction for RM=5:", ridge_pred_5)
print("Lasso Prediction for RM=5:", lasso_pred_5)
