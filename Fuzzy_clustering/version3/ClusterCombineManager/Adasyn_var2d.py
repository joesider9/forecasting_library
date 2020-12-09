"""Class to perform over-sampling using ADASYN."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
from scipy import sparse

from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_neighbors_object
from imblearn.utils import check_sampling_strategy



class ADASYN():


    def __init__(
        self,
        sampling_strategy="auto",
        variables=None,
        variables_3d=None,
        random_state=None,
        n_neighbors=5,
        n_jobs=None,
    ):
        self.sampling_strategy=sampling_strategy
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.variables = variables
        self.variables_3d = variables_3d


    def _validate_estimator(self):
        """Create the necessary objects for ADASYN"""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1
        )
        self.nn_.set_params(**{"n_jobs": self.n_jobs})

    def fit_resample(self, X, X_3d, y, y_org):
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, 'over-sampling'
        )
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = [X.copy()]
        X_3d_resampled = [X_3d.copy()]
        y_resampled = [y.copy()]
        y_org_resampled = [y_org.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)
            X_class_3d = _safe_indexing(X_3d, target_class_indices)
            y_class_org = _safe_indexing(y_org, target_class_indices)

            # self.nn_.set_params(**{"n_neighbors": self.n_neighbors})
            self.nn_.fit(X[:, self.variables])
            nns = self.nn_.kneighbors(X_class[:, self.variables], return_distance=False)[:, 1:]
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            n_neighbors = self.nn_.n_neighbors - 1
            ratio_nn = np.sum(y[nns] != class_sample, axis=1) / n_neighbors
            if not np.sum(ratio_nn):
                raise RuntimeError(
                    "Not any neigbours belong to the majority"
                    " class. This case will induce a NaN case"
                    " with a division by zero. ADASYN is not"
                    " suited for this specific dataset."
                    " Use SMOTE instead."
                )
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            # rounding may cause new amount for n_samples
            n_samples = np.sum(n_samples_generate)
            if not n_samples:
                raise ValueError(
                    "No samples will be generated with the"
                    " provided ratio settings."
                )

            # the nearest neighbors need to be fitted only on the current class
            # to find the class NN to generate new samples
            # self.nn_.set_params(**{"n_neighbors": np.minimum(int(X_class.shape[0]-1), self.n_neighbors)})
            self.nn_.fit(X_class[:, self.variables])
            nns = self.nn_.kneighbors(X_class[:, self.variables], return_distance=False)[:, 1:]

            enumerated_class_indices = np.arange(len(target_class_indices))
            rows = np.repeat(enumerated_class_indices, n_samples_generate)
            cols = random_state.choice(n_neighbors, size=n_samples)
            diffs = X_class[nns[rows, cols]][:, self.variables] - X_class[rows][:, self.variables]
            diffs_3d = X_class_3d[nns[rows, cols]][:, self.variables_3d, :] - X_class_3d[rows][:, self.variables_3d, :]
            steps = random_state.uniform( size=(n_samples, 1))
            X_new = X_class[rows]
            X_new_3d = X_class_3d[rows]
            y_new_org = y_class_org[rows]

            if sparse.issparse(X):
                sparse_func = type(X).__name__
                steps = getattr(sparse, sparse_func)(steps)
                X_new[:, self.variables] = X_class[rows][:, self.variables] + steps.multiply(diffs)
                X_new_3d[:, self.variables_3d, :] = X_class_3d[rows][:, self.variables_3d, :] + steps[:, :,
                                                                                                np.newaxis].multiply(diffs)
            else:
                X_new[:, self.variables] = X_class[rows][:, self.variables] + steps * diffs
                X_new_3d[:, self.variables_3d, :] = X_class_3d[rows][:, self.variables_3d, :] + steps[:, :,
                                                                                                np.newaxis] * diffs_3d

            X_new = X_new.astype(X.dtype)
            X_new_3d = X_new_3d.astype(X.dtype)
            y_new = np.full(n_samples, fill_value=class_sample, dtype=y.dtype)
            X_resampled.append(X_new)
            X_3d_resampled.append(X_new_3d)
            y_resampled.append(y_new)
            y_org_resampled.append(y_new_org)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
            X_3d_resampled = sparse.vstack(X_3d_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
            X_3d_resampled = np.vstack(X_3d_resampled)
        y_resampled = np.hstack(y_resampled)
        y_org_resampled = np.hstack(y_org_resampled)

        return X_resampled, X_3d_resampled, y_org_resampled
