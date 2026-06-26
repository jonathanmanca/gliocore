"""
segmentation/mrf_em.py — Markov Random Field con EM, multivariato.

GMM multivariato + prior spaziale di Ising (ICM).
Funziona su feature N×D. Il prior spaziale usa il volume 3D per la connettività.
"""
from __future__ import annotations
import logging
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext

log = logging.getLogger(__name__)

_NEIGHBORS_6 = np.array([
    [-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]
], dtype=np.int32)


class MRFEMSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        beta: float = 1.5,
        k: int = 3,
        max_iter: int = 20,
        tol: float = 0.001,
        n_init: int = 5,
        covariance_type: str = "full",
    ):
        self.beta = beta
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.covariance_type = covariance_type

    @property
    def name(self) -> str:
        return "MRF-EM"

    @property
    def description(self) -> str:
        return (
            "Multivariate Markov Random Field: GMM + Ising spatial prior. "
            "Reduces isolated voxels, produces anatomically coherent maps. "
            f"β={self.beta} controls the strength of the spatial prior."
        )

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        X = features.matrix
        mask = features.mask
        n = len(X)

        # Inizializzazione GMM multivariato
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gmm = GaussianMixture(
                n_components=self.k, covariance_type=self.covariance_type,
                n_init=self.n_init, random_state=42,
            )
            gmm.fit(X)

        labels = gmm.predict(X)   # 0-indexed
        log.info(f"MRF-EM [{context.modality.value}]: k={self.k}, β={self.beta}, "
                 f"{features.n_features} features")

        # Volume etichette per ICM
        labels_vol = np.zeros(mask.shape, dtype=np.int32)
        labels_vol[mask] = labels
        coords = np.column_stack(np.where(mask))

        means = gmm.means_           # (k, D)
        covs  = gmm.covariances_
        weights = gmm.weights_

        n_changed_history = []
        for iteration in range(self.max_iter):
            # M-step: aggiorna parametri
            means, covs_diag, weights = self._m_step(X, labels, self.k)
            # ICM-step
            new_labels = self._icm_step(
                X, labels_vol, coords, mask, means, covs_diag, weights, self.k
            )
            n_changed = int((new_labels != labels).sum())
            n_changed_history.append(n_changed)
            labels = new_labels
            labels_vol[mask] = labels
            if n_changed / n < self.tol:
                log.info(f"MRF-EM converged at iter {iteration+1}")
                break

        extra = {
            "best_k": self.k,
            "beta": self.beta,
            "iterations": len(n_changed_history),
            "converged": len(n_changed_history) < self.max_iter,
        }
        return labels, None, extra

    def _m_step(self, X, labels, k):
        D = X.shape[1]
        means = np.zeros((k, D))
        variances = np.ones((k, D)) * 0.01
        weights = np.zeros(k)
        for c in range(k):
            idx = labels == c
            nc = idx.sum()
            if nc == 0:
                continue
            weights[c] = nc / len(X)
            means[c] = X[idx].mean(axis=0)
            variances[c] = np.maximum(X[idx].var(axis=0), 1e-6)
        return means, variances, weights

    def _icm_step(self, X, labels_vol, coords, mask, means, variances, weights, k):
        shape = labels_vol.shape
        new_labels = labels_vol[mask].copy()
        # Precompute log-likelihood diagonale per ogni voxel/cluster
        for i, (z, y, x) in enumerate(coords):
            xi = X[i]
            best_label, best_energy = new_labels[i], np.inf
            for c in range(k):
                # Data term: gaussiana diagonale multivariata
                diff = xi - means[c]
                data_e = 0.5 * np.sum(diff**2 / variances[c]) \
                         + 0.5 * np.sum(np.log(variances[c]))
                data_e -= np.log(max(weights[c], 1e-10))
                # Spatial term
                n_disc = 0
                for dz, dy, dx in _NEIGHBORS_6:
                    nz, ny, nx = z+dz, y+dy, x+dx
                    if (0 <= nz < shape[0] and 0 <= ny < shape[1]
                            and 0 <= nx < shape[2] and mask[nz, ny, nx]):
                        if labels_vol[nz, ny, nx] != c:
                            n_disc += 1
                energy = data_e + self.beta * n_disc
                if energy < best_energy:
                    best_energy, best_label = energy, c
            new_labels[i] = best_label
        return new_labels

    def _get_params(self) -> dict:
        return {"beta": self.beta, "k": self.k, "max_iter": self.max_iter}
