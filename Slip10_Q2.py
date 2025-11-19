import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("iris.csv")

le = LabelEncoder()
df["species_num"] = le.fit_transform(df["species"])

plt.scatter(df["sepal_length"], df["sepal_width"], c=df["species_num"])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Scatter Plot")
plt.show()
