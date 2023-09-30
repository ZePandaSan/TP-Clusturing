import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Reload the CSV with correct delimiter
planete = pd.read_csv('Data/planete.csv', delimiter=';')

# Remove the last column
planete = planete.iloc[:, :-1]

# Display the first few rows of the DataFrame without labels
print(planete.head())



# Initialize lists to store the values of the indices for each number of clusters
davies_bouldin_values = []
calinski_harabasz_values = []

# Define the range of number of clusters
cluster_range = range(2, 11)  # usually, start from 2 because 1 cluster is trivial and not meaningful

# For each number of clusters, fit the data, predict the labels and calculate the indices
for n_clusters in cluster_range:
    clusterer = KMeans(n_clusters=n_clusters, random_state=42).fit(planete)
    cluster_labels = clusterer.predict(planete)
    
    # Calculate the Davies Bouldin score
    davies_bouldin_val = davies_bouldin_score(planete, cluster_labels)
    davies_bouldin_values.append(davies_bouldin_val)
    
    # Calculate the Calinski Harabasz score
    calinski_harabasz_val = calinski_harabasz_score(planete, cluster_labels)
    calinski_harabasz_values.append(calinski_harabasz_val)

# Plot the Davies Bouldin scores
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cluster_range, davies_bouldin_values, marker='o')
plt.title("Davies-Bouldin Index vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Davies-Bouldin Index")

# Plot the Calinski Harabasz scores
plt.subplot(1, 2, 2)
plt.plot(cluster_range, calinski_harabasz_values, marker='o')
plt.title("Calinski-Harabasz Index vs. Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("Calinski-Harabasz Index")

plt.tight_layout()
plt.savefig("Data/planete_davies_calinski.png")
plt.show()




# Execute k-means clustering with the optimal number of clusters (k=3 for this example)
optimal_clusters = 3
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42).fit(planete)
labels = kmeans.labels_

# Perform PCA and reduce the data to 2 dimensions
pca = PCA(n_components=2)
principal_components = pca.fit_transform(planete)

# Plot the clusters in the 2D PCA space
plt.figure(figsize=(10, 8))
for i in range(optimal_clusters):
    plt.scatter(principal_components[labels == i, 0], principal_components[labels == i, 1], label=f'Cluster {i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.title('2D PCA of Planete Data with Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.savefig("Data/planete_pca_3cluster.png")
plt.show()

plt.figure(figsize=(10, 8))
for i in range(optimal_clusters):
    plt.scatter(principal_components[labels == i, 0], principal_components[labels == i, 1], label=f'Cluster {i+1}')
plt.title('2D PCA of Planete Data with Cluster Labels')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()

