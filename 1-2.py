# Import des librairies numpy et preprocessing
import numpy as np
from sklearn import preprocessing
import csv
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


def csv_to_matrix(filename):
    """
    Convert a CSV file to a numpy matrix using the csv module.
    
    Parameters:
    - filename (str): Path to the CSV file.
    
    Returns:
    - numpy.matrix: Numpy matrix representation of the CSV data.
    """
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(list(map(float, row)))
            
    return np.matrix(data)




def matrix_to_csv(matrix, filename):
    """
    Convert a numpy matrix to a CSV file using pandas with comma and space delimiter.
    
    Parameters:
    - matrix (numpy.matrix or numpy.ndarray): Numpy matrix or array to be saved.
    - filename (str): Path to save the CSV file.
    
    Returns:
    - None: The matrix is saved as a CSV at the specified filename.
    """
    df = pd.DataFrame(matrix)
    df.to_csv(filename, sep=',', index=False, header=False)
    return None










def meanAndVariance(matrix):
    """
    Compute the mean and variance of each column of a matrix.
    
    Parameters:
    - matrix (numpy.matrix or numpy.ndarray): Numpy matrix or array to be saved.
    
    Returns:
    - A tuple containing the mean and variance of the matrix.
    """
    return (np.mean(matrix), np.var(matrix))


def normalize_matrix(matrix):
    """
    Normalize (center and scale) a numpy matrix or array and return as numpy matrix.
    
    Parameters:
    - matrix (numpy.matrix or numpy.ndarray): The matrix or array to be normalized.
    
    Returns:
    - numpy.matrix: The normalized matrix.
    """
    # Convert the matrix to a numpy array
    array_data = np.asarray(matrix)
    
    # Use the scale function to normalize
    normalized_array = preprocessing.scale(array_data)
    
    # Convert the normalized array back to a matrix
    return np.matrix(normalized_array)

def minMaxScal(matrix):
    """
    Normalize (center and scale) a numpy matrix or array and return as numpy matrix.
    
    Parameters:
    - matrix (numpy.matrix or numpy.ndarray): The matrix or array to be normalized.
    
    Returns:
    - numpy.matrix: The normalized matrix.
    """
    # Convert the matrix to a numpy array
    array_data = np.asarray(matrix)
    
    # Use the scale function to normalize
    normalized_array = preprocessing.MinMaxScaler().fit_transform(array_data)
    
    # Convert the normalized array back to a matrix
    return np.matrix(normalized_array)


def read__data(filename):
    """
    Lit le fichier iris.data et retourne les données dans une DataFrame pandas.

    Parameters:
    - chemin_du_fichier (str): Le chemin vers le fichier iris.data.

    Returns:
    - pd.DataFrame: Les données iris dans une DataFrame pandas.
    """
    # Lire le fichier iris.data en utilisant la première ligne comme en-tête
    iris_data = pd.read_csv(filename, header=0)
    return iris_data

def perform_pca(data, n_components=None):
    """
    Perform Principal Component Analysis (PCA) on the input data.

    Parameters:
    - data (pd.DataFrame or np.ndarray): The input data to perform PCA on.
    - n_components (int, optional): The number of principal components to retain. If None, all components are retained.

    Returns:
    - pd.DataFrame: The transformed data in the reduced dimensionality.
    - PCA object: The PCA model that can be used for further analysis or transformation.
    """
    # Initialize PCA with the desired number of components
    pca = PCA(n_components=n_components)
    
    # Fit PCA on the data and transform it
    transformed_data = pca.fit_transform(data)
    
    # Create a DataFrame from the transformed data
    transformed_df = pd.DataFrame(data=transformed_data, columns=[f"PC{i+1}" for i in range(transformed_data.shape[1])])
    
    return transformed_df, pca



def plot_pca_2d(transformed_data, species_labels, save_path):
    """
    Create a 2D PCA visualization of transformed data.

    Parameters:
    - transformed_data (pd.DataFrame): DataFrame containing transformed data after PCA.
    - species_labels (pd.Series or np.ndarray): Labels of species for coloring points.

    Returns:
    - None: Displays the 2D PCA plot.
    """
    # Extract the two first principal components
    pc1 = transformed_data["PC1"]
    pc2 = transformed_data["PC2"]

    # Define a color map for species labels
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

    # Map species labels to colors
    colors = np.array([colormap[label] for label in species_labels])

    # Create a 2D scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, pc2, c=colors)  # Color points based on species labels
    plt.title("PCA Visualization in 2D")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.savefig(save_path, format="png")



def plot_pca_3d(transformed_data, species_labels):
    """
    Create a 3D PCA visualization of transformed data.

    Parameters:
    - transformed_data (pd.DataFrame): DataFrame containing transformed data after PCA.
    - species_labels (pd.Series or np.ndarray): Labels of species for coloring points.

    Returns:
    - None: Displays the 3D PCA plot.
    """
    # Extract the three first principal components
    pc1 = transformed_data["PC1"]
    pc2 = transformed_data["PC2"]
    pc3 = transformed_data["PC3"]

    # Define a color map for species labels
    colormap = {'setosa': 'red', 'versicolor': 'green', 'virginica': 'blue'}

    # Map species labels to colors
    colors = np.array([colormap[label] for label in species_labels])

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pc1, pc2, pc3, c=colors)  # Color points based on species labels
    ax.set_title("PCA Visualization in 3D")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    plt.colorbar(scatter, label="Species")
    plt.show()

def kmeans_clustering_and_plot(data_pca, num_clusters, save_path="kmeans.png"):
    """
    Perform K-Means clustering on PCA data and visualize the results with centroids.

    Parameters:
    - data_pca (numpy.ndarray): The PCA-transformed dataset.
    - num_clusters (int): The number of clusters to form.

    Returns:
    - centroides (numpy.ndarray): The centroids of the clusters.
    """

    # Create and fit the K-Means model with the specified number of clusters
    kmeans_model = KMeans(n_clusters=num_clusters)
    kmeans_model.fit(data_pca)

    # Get cluster labels and centroids
    cluster_labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=cluster_labels, cmap='viridis', s=50)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.title(f'K-Means Clustering ({num_clusters} Clusters)')
    plt.savefig(save_path)
    return centroids

# 1- Normalisation 
print("1- Normalisation :\n")
matrix_X=csv_to_matrix("Data/matrix_X.csv")
print("Visualisation de la matrice X :")
print(matrix_X)
meanVariance_X=meanAndVariance(matrix_X)
print("Moyenne et variance de la matrice X : ")
print(meanVariance_X)
normalized_X=normalize_matrix(matrix_X)
print("Matrice X normalisée : ")
print(normalized_X)
matrix_to_csv(normalized_X,"Data/normalized_X.csv")
meanVariance_normalized_X=meanAndVariance(normalized_X)
print("Moyenne et variance de la matrice X normalisée : ")
print(meanVariance_normalized_X)

# 2- MinMaxScaler
print("\n2- MinMaxScaler :\n")
matrix_X2=csv_to_matrix("Data/matrix_X2.csv")
print("Visualisation de la matrice X2 :")
print(matrix_X2)
meanAndVariance_X2=meanAndVariance(matrix_X2)
print("Moyenne et variance de la matrice X2 : ")
print(meanAndVariance_X2)
normalized_X2=minMaxScal(matrix_X2)
print("Matrice X2 normalisée : ")
print(normalized_X2)
matrix_to_csv(normalized_X2,"Data/normalized_X2.csv")
meanVariance_normalized_X2=meanAndVariance(normalized_X2)
print("Moyenne et variance de la matrice X2 normalisée : ")
print(meanVariance_normalized_X2)

# 3- K-means
#print("\n3- K-means :\n")
#iris=read__data("Data/iris.data")
#print("Visualisation des données iris :")
#print(iris)
#species_labels = iris.pop("Class")
#print(species_labels.head())
#print(iris.head())
#transformed_data, pca_model = perform_pca(iris, n_components=2)
#print(transformed_data.head())
#transformed_data.to_csv("Data/transformed_iris_2cmp.csv", index=False)
#plot_pca_2d(transformed_data, species_labels, "Data/iris_pca_2d.png")
#transformed_data, pca_model = perform_pca(iris, n_components=3)
#transformed_data.to_csv("Data/transformed_iris_3cmp.csv", index=False)
#plot_pca_3d(transformed_data, species_labels)
#data_transformed_numpy = transformed_data.to_numpy()
#list_centroids=kmeans_clustering_and_plot(data_transformed_numpy, 3, save_path="Data/iris_kmeans_3clust.png")



# Suite du code : 3.py




