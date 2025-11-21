import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

data = pd.read_csv('iris.csv')  

X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = data['species']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_pca, y_encoded, test_size=0.2, random_state=42)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

def predict_flower(sepal_length, sepal_width, petal_length, petal_width):
    new_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    new_scaled = scaler.transform(new_data)
    new_pca = pca.transform(new_scaled)
    pred = model.predict(new_pca)
    return le.inverse_transform(pred)[0]

flower_type = predict_flower(5.1, 3.5, 1.4, 0.2)
print("Predicted Flower Type:", flower_type)
