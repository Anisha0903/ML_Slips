import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("diabetes.csv")

scaler = StandardScaler()
scaled = scaler.fit_transform(data)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled)

labels = kmeans.labels_

print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Labels:\n", labels)
