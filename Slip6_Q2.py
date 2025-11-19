import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('employees.csv')
dataset = dataset.dropna()

X = dataset[['Annual_Income']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

k = 3
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
dataset['Income_Group'] = y_kmeans

plt.figure(figsize=(8,5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(k):
    plt.scatter(X_scaled[y_kmeans == i, 0], [0]*sum(y_kmeans == i), s=100, c=colors[i], label=f'Cluster {i+1}')
plt.xlabel('Scaled Annual Income')
plt.title('Employee Income Clusters')
plt.legend()
plt.show()

print(dataset.head())
