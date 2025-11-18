import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

df = pd.read_csv("wholesale_customers.csv")

X = df[["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicassen"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

cluster = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward")
labels = cluster.fit_predict(X_scaled)

df["Cluster"] = labels
print(df[["Region","Cluster"]])

plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.xlabel("Fresh (scaled)")
plt.ylabel("Milk (scaled)")
plt.title("Agglomerative Clustering - Wholesale Customers")
plt.show()
