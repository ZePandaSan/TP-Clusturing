from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Load the data from the uploaded file
iris_data = pd.read_csv('Data/iris.data', header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])


# Remove the first row with repeated headers
iris_data = iris_data.drop(0)

# Convert numeric columns to float
numeric_columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
iris_data[numeric_columns] = iris_data[numeric_columns].astype(float)

# Display the cleaned data
iris_data.head()



# Extracting the features for clustering
X = iris_data[numeric_columns]

# Applying Agglomerative Clustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
iris_data['cluster_labels'] = cluster.fit_predict(X)

# Visualizing the clusters using the first two features (sepal_length and sepal_width)
plt.figure(figsize=(10, 7))
plt.scatter(iris_data['sepal_length'], iris_data['sepal_width'], c=iris_data['cluster_labels'], cmap='rainbow')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Clusters based on Sepal Length and Sepal Width')
plt.colorbar(label='Cluster Label')
plt.savefig("Data/iris_agglomerative_3clust_sepal.png")
plt.show()



# Recompute the linkage matrix using "complete" linkage
linked = linkage(X, 'complete')
threshold_distance = 3.5
# Plot the dendrogram again
plt.figure(figsize=(15, 7))
dendrogram(linked, orientation='top', color_threshold=threshold_distance, distance_sort='descending', show_leaf_counts=True)
plt.axhline(y=threshold_distance, color='r', linestyle='--')
plt.title('Dendrogram with Complete Linkage Highlighting 3 Main Clusters')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.savefig("Data/iris_dendrogram_complete.png")
plt.show()

from scipy.cluster.hierarchy import fcluster

# Form 3 clusters using the linkage matrix
labels = fcluster(linked, t=threshold_distance, criterion='distance')

# Visualize the clusters
plt.figure(figsize=(10, 7))
plt.scatter(iris_data['sepal_length'], iris_data['sepal_width'], c=labels, cmap='rainbow')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Clusters based on Sepal Length and Sepal Width')
plt.colorbar(label='Cluster Label')
plt.savefig("Data/iris_agglomerative_3clust_sepal_complete.png")
plt.show()
# Create a contingency table comparing the obtained cluster labels with the actual iris class labels
contingency_table = pd.crosstab(iris_data['class'], labels, rownames=['Actual'], colnames=['Predicted'])

print(contingency_table)

# Indice de silhouette
print("Indice de silhouette :")
from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(X, labels)
print(silhouette_score)


# Compute the linkage matrix using "average" linkage
linked_average = linkage(X, 'average')

# Plot the dendrogram for "average" linkage
plt.figure(figsize=(15, 7))
dendrogram(linked_average, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram with Average Linkage')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.savefig("Data/iris_dendrogram_average.png")
plt.show()

# Visualize the clusters obtained with average linkage
plt.figure(figsize=(10, 7))
plt.scatter(iris_data['sepal_length'], iris_data['sepal_width'], c=labels, cmap='rainbow')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Clusters based on Sepal Length and Sepal Width (Average Linkage)')
plt.colorbar(label='Cluster Label')
plt.show()



# Form 3 clusters using the linkage matrix with average linkage
labels_average = fcluster(linked_average, t=1.5, criterion='distance')

# Create a contingency table comparing the obtained cluster labels with the actual iris class labels for average linkage
contingency_table_average = pd.crosstab(iris_data['class'], labels_average, rownames=['Actual'], colnames=['Predicted'])

print(contingency_table_average)

# Indice de silhouette
print("Indice de silhouette :")
from sklearn.metrics import silhouette_score
silhouette_score = silhouette_score(X, labels_average)
print(silhouette_score)










