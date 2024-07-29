import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Set the number of points
num_points = np.random.randint(100, 400)  # n >= 100, up to 200 for variety
points = np.array([(np.random.randint(-100, 100), np.random.randint(-100, 100)) for _ in range(num_points)])

# Randomly choose k between 2 and 5
k = np.random.randint(2, 6)

# Initialize centroids
centroids = [points[np.random.randint(0, num_points)] for _ in range(k)]


def k_means(points, centroids, iterations):
    for j in range(iterations):
        clusters = {i: [] for i in range(k)}
        for point in points:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = distances.index(min(distances))
            clusters[cluster_index].append(point)

        for i in range(k):
            if clusters[i]:
                centroids[i] = np.mean(clusters[i], axis=0)

        if j == 0:
            plot_clusters(clusters, centroids, 'First Iteration')
            plot_3d_clusters(clusters, centroids, 'First Iteration')

        elif j == iterations // 2:
            plot_clusters(clusters, centroids, 'Middle Iteration')
            plot_3d_clusters(clusters, centroids, 'Middle Iteration')

    return clusters, centroids

colors = ['red', 'blue', 'green', 'yellow', 'black']

def plot_3d_clusters(clusters, centroids, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i, (cluster_idx, cluster_points) in enumerate(clusters.items()):
        if cluster_points:
            cluster_points = np.array(cluster_points)
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], np.zeros(cluster_points.shape[0]))
            ax.scatter(centroids[i][0], centroids[i][1],0, c='purple', marker='x')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.show()



def plot_clusters(clusters, centroids, title):
    colors = ['red', 'blue', 'green', 'yellow', 'black']
    for i, (cluster_idx, cluster_points) in enumerate(clusters.items()):
        if cluster_points:
            cluster_points = np.array(cluster_points)
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Cluster {i}')
            plt.scatter(centroids[i][0], centroids[i][1], c='purple', marker='x')
    plt.legend(fontsize=8)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.show()


iterations = 10

clusters1, centroids1 = k_means(points, centroids, iterations)

plot_3d_clusters(clusters1, centroids, '3D View')


plot_clusters(clusters1, centroids1, 'Last Iteration')
