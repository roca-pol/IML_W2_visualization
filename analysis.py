import numpy as np
import pandas as pd

def _sanitize(X):
    return X.to_numpy() if isinstance(X, pd.DataFrame) else X


class PCA:
    
    def __init__(self, n_components, verbose=True):
        self.n_components = n_components
        self.verbose = verbose

    def fit(self, X):
        X = _sanitize(X)

        covmat = np.cov(X.T)
        eigvals, eigvecs = np.linalg.eig(covmat)

        if self.verbose:
            print('[PCA] Covariance matrix:\n', covmat)
            print('\n[PCA] Eigenvalues:\n', eigvals)
            print('\n[PCA] Eigenvectors:\n', eigvecs)

        # convert to positive since we want the highest absolute value
        eigvecs[:, eigvals < 0] *= -1.0
        eigvals[eigvals < 0] *= -1.0

        # sort eigvals since they might not be ordered
        ordered_idxs = np.argsort(eigvals)[::-1]    # -1 to obtain descending order
        eigvals = eigvals[ordered_idxs]
        eigvecs = eigvecs[:, ordered_idxs]

        if self.verbose:
            print('\n[PCA] Eigenvalues (sorted):\n', eigvals)
            print('\n[PCA] Eigenvectors (sorted):\n', eigvecs, '\n')

        self.mean_ = X.mean(axis=0)
        self.eigvals_ = eigvals[:self.n_components]
        self.eigvecs_ = eigvecs[:, :self.n_components]
        return self

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def transform(self, X):
        X = _sanitize(X)

        X_adj = X - self.mean_
        X_trans = np.dot(X_adj, self.eigvecs_)

        return X_trans

    def inverse_transform(self, X):
        X = _sanitize(X)

        X_adj = np.dot(X, self.eigvecs_.T)
        X_orig = X_adj + self.mean_

        return X_orig
