import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

iris = pd.read_csv("iris.csv")

le = LabelEncoder()
iris["species_num"] = le.fit_transform(iris["species"])

plt.scatter(iris["sepal_length"], iris["sepal_width"], c=iris["species_num"])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset Scatter Plot")
plt.show()
