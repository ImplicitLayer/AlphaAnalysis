import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import squareform


class PortfolioClustering:
    def __init__(self, returns):
        """
        A class for clustering assets in a portfolio based on their returns.

        :param returns: DataFrame with asset returns (assets - columns, dates - rows).
        """
        self.returns = returns
        self.correlation_matrix = returns.corr()

    def kmeans_clustering(self, num_clusters=3):
        """
        Clustering by K-Means method.

        :param num_clusters: Number of clusters.
        :return: DataFrame with clusters for assets.
        """
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(self.returns.T)

        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(returns_scaled)

        return pd.DataFrame({'Asset': self.returns.columns, 'Cluster': clusters})

    def hierarchical_clustering(self, method='ward'):
        """
        Hierarchical clustering.

        :param method: Agglomeration method (ward, single, complete, average).
        :return: dendrogram plot.
        """
        distance_matrix = squareform(1 - self.correlation_matrix)
        linkage_matrix = sch.linkage(distance_matrix, method=method)

        plt.figure(figsize=(10, 5))
        sch.dendrogram(linkage_matrix, labels=self.returns.columns, leaf_rotation=90)
        plt.title(f'Hierarchical Clustering ({method} method)')
        plt.show()

    def dbscan_clustering(self, eps=0.5, min_samples=2):
        """
        Clustering by DBSCAN method.

        :param eps: Neighborhood radius parameter.
        :param min_samples: Minimum number of points in the neighborhood.
        :return: DataFrame with clusters.
        """
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(self.returns.T)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(returns_scaled)

        return pd.DataFrame({'Asset': self.returns.columns, 'Cluster': clusters})
