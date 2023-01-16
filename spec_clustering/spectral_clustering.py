import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()


def spectral_clustering(data, n_cl, sigma=1., fiedler_solution=False):
    """
    Spectral clustering.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    sigma: float
        std of radial basis function kernel.
    fiedler_solution: bool
        return fiedler solution instead of kmeans

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    if fiedler_solution and n_cl != 2:
        raise Exception("Cannot apply Fiedler to more than 2 clusters!")

    # compute distances
    dist_matrix = ((np.expand_dims(data, 0) -
                    np.expand_dims(data, 1)) ** 2).sum(-1)
    # compute affinity matrix
    affinity_matrix = np.exp(-dist_matrix / (sigma ** 2))

    # compute degree matrix
    degree_matrix = np.diag(affinity_matrix.sum(1))

    # compute laplacian
    laplacian_matrix = degree_matrix-affinity_matrix

    # compute eigenvalues and vectors
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)
    # ensure we are not using complex numbers - you shouldn't btw
    if eigenvalues.dtype == 'complex128':
        print(
            "We, gli autovalori sono complessi. Ora, io non mi rompo, ma dai un po' su a questo sigma (σ). (;")
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and vectors
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues, eigenvectors = eigenvalues[sorted_indices], eigenvectors[:, sorted_indices]

    # Fiedler-vector solution
    labels = eigenvectors[:, 1] > 0
    if fiedler_solution:
        return labels

    # KMeans solution
    new_features = eigenvectors[:, 1:n_cl+1]
    labels = KMeans(n_cl).fit_predict(new_features)

    return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    # data, cl = two_moon_dataset(n_samples=300, noise=0.1)  # best sigma = 0.1
    data, cl = gaussians_dataset(n_gaussian=3, n_points=[100, 100, 70], mus=[[1,1], [-4,6], [8,8]], stds=[[1,1], [3,3],[1,1]]) # best sigma = 2

    # visualize the dataset
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    # plt.waitforbuttonpress()

    # run spectral clustering
    labels = spectral_clustering(data, n_cl=2, sigma=0.1)

    # visualize results
    ax[1].scatter(data[:, 0], data[:, 1], c=labels, s=40)
    plt.waitforbuttonpress()


if __name__ == '__main__':
    main_spectral_clustering()
