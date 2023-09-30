import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Reload the original iris data, skipping the first row
iris_data_original = pd.read_csv("Data/iris.data", header=None, skiprows=1)
iris_data_original.columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Class"]

# Extract species labels again
species_labels_original = iris_data_original.pop("Class")

# Performing KMeans clustering with 3 clusters on the corrected original data
kmeans_model_original = KMeans(n_clusters=3, n_init=10)
kmeans_model_original.fit(iris_data_original)
cluster_labels_original = kmeans_model_original.labels_
centroids_original = kmeans_model_original.cluster_centers_


# Visualizing the KMeans clusters vs True Labels on the original data
plt.figure(figsize=(10, 6))

# KMeans Clusters
scatter_kmeans = plt.scatter(iris_data_original["PetalLength"], iris_data_original["PetalWidth"], c=cluster_labels_original, cmap='viridis', s=50, alpha=0.6, label="KMeans Clusters")
plt.scatter(centroids_original[:, 2], centroids_original[:, 3], c='red', marker='X', s=200)

# True Labels
color_map_original = {"setosa": 0, "versicolor": 1, "virginica": 2}
colors_original = [color_map_original[label] for label in species_labels_original]
scatter_true = plt.scatter(iris_data_original["PetalLength"], iris_data_original["PetalWidth"], c=colors_original, cmap='jet', s=15, marker='x', label="True Labels")

# Legend
legend1 = plt.legend(handles=[scatter_kmeans, scatter_true], loc="upper right")
plt.gca().add_artist(legend1)

# Custom legend for species
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Setosa', markersize=10, markerfacecolor='blue'),
                   Line2D([0], [0], marker='o', color='w', label='Versicolor', markersize=10, markerfacecolor='cyan'),
                   Line2D([0], [0], marker='o', color='w', label='Virginica', markersize=10, markerfacecolor='yellow')]
plt.legend(handles=legend_elements, loc="lower right")

plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('KMeans Clustering vs True Labels on Original Data')
plt.savefig("Data/iris_kmeans_3clust_vs_true_original.png")
plt.show()


# Create the contingency table for clustering on original data
contingency_table_original = pd.crosstab(cluster_labels_original, species_labels_original, rownames=['Cluster'], colnames=['Species'])

print(contingency_table_original)

# Calculate the silhouette score for the KMeans clustering on original data
silhouette_avg_original = silhouette_score(iris_data_original, cluster_labels_original)

print(silhouette_avg_original)


