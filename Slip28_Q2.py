import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

data = pd.read_csv('iris.csv')  

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Kernel: {kernel}, Accuracy: {acc:.2f}")

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    model = SVC(kernel='rbf')
    model.fit(X_train, y_train)
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return le.inverse_transform(prediction)[0]

flower_type = predict_flower(5.1, 3.5, 1.4, 0.2)
print("Predicted Flower Type:", flower_type)
