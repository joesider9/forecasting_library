import numpy as np


def statistical_guess(x, q):
    """Produces initial guesses for the cluster centers.

    The guesses are distributed within a box

    Parameters
    ----------
    x : ndarray, shape(n_samples, n_variables)
        Unlabeled object data.
    q : int
        Number of clusters to find.
    Returns
    -------
    c : (q, n_variables), ndarray
        Cluster centers.

    """
    c = 2 * np.std(x, axis=0) * (np.random.random((q, x.shape[1]))-0.5) + np.mean(x, axis=0)
    return c
