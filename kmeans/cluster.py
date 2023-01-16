import numpy as np

from numpy.random import uniform
from numpy.matlib import repmat

import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, n_cl: int, n_init : int =  1,
                 initial_centers = None, verbose: bool = False):
        """
        Parameters
        ----------
        n_cl: int
            number of clusters.
        n_init: int
            number of time the k-means algorithm will be run.
        initial_centers:
            If an ndarray is passed, it should be of shape (n_clusters, n_features) 
            and gives the initial centers.
        verbose: bool
            whether or not to plot assignment at each iteration (default is True).
        """
        
        self.n_cl = n_cl
        self.n_init = n_init
        self.initial_centers = initial_centers
        self.verbose = verbose

    def _init_centers(self, X: np.array, use_samples: bool = False):

        n_samples, dim = X.shape #Returns the shape rows * cols of X

        if use_samples:
            return X[np.random.choice(n_samples, size=self.n_cl)] # For K-Medoids

        centers = np.zeros((self.n_cl, dim)) # Initialize array of centers to 0
        for i in range(dim): 
            min_f, max_f = np.min(X[:, i]), np.max(X[:, i])
            centers[:, i] = uniform(low=min_f, high=max_f, size=self.n_cl)
        return centers

    def single_fit_predict(self, X: np.array):
        """
        Kmeans algorithm.

        Parameters
        ----------
        X: ndarray
            data to partition, Expected shape (n_samples, dim).
        
        Returns
        -------
        centers: ndarray
            computed centers. Expected shape (n_cl, dim)
        
        assignment: ndarray
            computed assignment. Expected shape (n_samples,)
        """

        n_samples, dim = X.shape

        # initialize centers
        centers = np.array(self._init_centers(X)) if self.initial_centers is None \
            else np.array(self.initial_centers)

        old_assignments = np.zeros(shape=n_samples)

        if self.verbose:
            fig, ax = plt.subplots()

        while True:  # stopping criterion

            if self.verbose:
                ax.scatter(X[:, 0], X[:, 1], c=old_assignments, s=40)
                ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
                ax.axis('off')
                plt.pause(1)
                plt.cla()

            # assign
            distances = np.zeros(shape=(n_samples, self.n_cl)) # Distances array
            for c_idx, c in enumerate(centers): #c_idx center index, c actual center
                distances[:, c_idx] = np.sum(np.square(X - repmat(c, n_samples, 1)), axis=1) #Compute the distances

            new_assignments = np.argmin(distances, axis=1) # Re-Assing the points to the center

            # re-estimate the center
            for l in range(0, self.n_cl):
                centers[l] = np.mean(X[new_assignments == l], axis=0)

            # If no points changed center, stop
            if np.all(new_assignments == old_assignments):
                break

            # update
            old_assignments = new_assignments

        if self.verbose:
            plt.close()

        return centers, new_assignments

    def compute_cost_function(self, X: np.array, centers: np.array, assignments: np.array):
        """
        Returns
        -------
        cost: float
            computed cost function.
        """
        
        cost = 0.0

        for i in range(self.n_cl):
            assigned_points = X[assignments == i]
            num_assigned_points = assigned_points.shape[0]
            cost += np.square(assigned_points - repmat(centers[i], num_assigned_points, 1)).sum()

        return cost

    def fit_predict(self, X: np.array):
        """
        Returns
        -------
        assignment: ndarray
            computed assignment. Expected shape (n_samples,)
        """
        
        assignments_opt = None
        cost_min = np.inf

        for i in range(self.n_init):
            centers, assignments = self.single_fit_predict(X)
            cost = self.compute_cost_function(X, centers, assignments)
            if self.verbose:
                print(f'Iteration: {i} - cost function: {cost}')
            if cost < cost_min:
                cost_min = cost
                assignments_opt = assignments

        return assignments_opt