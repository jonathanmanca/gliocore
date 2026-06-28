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
        """
        Un passo ICM vettorizzato, equivalente bit-a-bit al ciclo scalare.

        L'aggiornamento è sincrono (Jacobi): l'energia spaziale di ogni voxel
        usa SOLO le etichette del passo precedente (`labels_vol`), mai quelle
        aggiornate nello stesso sweep. Non essendoci dipendenze intra-sweep, il
        calcolo per-voxel è indipendente e si vettorizza senza cambiare i valori.
        `coords` è mantenuto per compatibilità di firma ma non più necessario.
        """
        # ── Data term (N, k): gaussiana diagonale multivariata ───────────────
        diff = X[:, None, :] - means[None, :, :]                 # (N, k, D)
        quad = 0.5 * np.sum(diff ** 2 / variances[None, :, :], axis=2)   # (N, k)
        log_var = 0.5 * np.sum(np.log(variances), axis=1)        # (k,)
        log_w = np.log(np.maximum(weights, 1e-10))               # (k,)
        e_data = quad + log_var[None, :] - log_w[None, :]        # (N, k)

        # ── Spatial term: n. di vicini (6-conn) con etichetta diversa ────────
        # n_disc[i, c] = (vicini validi di i) - (vicini di i con etichetta == c)
        mask_f = mask.astype(np.float64)
        onehot = np.zeros(mask.shape + (k,), dtype=np.float64)
        for c in range(k):
            onehot[..., c] = mask_f * (labels_vol == c)

        valid_count = np.zeros(mask.shape, dtype=np.float64)     # (Z, Y, X)
        same_count = np.zeros(mask.shape + (k,), dtype=np.float64)
        for off in _NEIGHBORS_6:
            valid_count += self._shift_lookahead(mask_f, off)
            same_count += self._shift_lookahead(onehot, off)
        n_disc_vol = valid_count[..., None] - same_count         # (Z, Y, X, k)
        n_disc = n_disc_vol[mask]                                # (N, k), C-order

        energy = e_data + self.beta * n_disc                     # (N, k)
        # argmin con tie-break sul primo indice = il `<` stretto dello scalare
        return np.argmin(energy, axis=1).astype(labels_vol.dtype)

    @staticmethod
    def _shift_lookahead(arr: np.ndarray, offset) -> np.ndarray:
        """
        out[p] = arr[p + offset] se in-bounds sui 3 assi spaziali, altrimenti 0.
        Supporta array 3D (Z,Y,X) e 4D (Z,Y,X,k); l'eventuale ultimo asse
        (canali) resta intatto. Niente wrap-around (a differenza di np.roll).
        """
        out = np.zeros_like(arr)

        def _slices(d, n):
            if d >= 0:
                return slice(0, n - d), slice(d, n)     # dest, src
            return slice(-d, n), slice(0, n + d)

        dz, dy, dx = (int(v) for v in offset)
        dest_z, src_z = _slices(dz, arr.shape[0])
        dest_y, src_y = _slices(dy, arr.shape[1])
        dest_x, src_x = _slices(dx, arr.shape[2])
        out[dest_z, dest_y, dest_x] = arr[src_z, src_y, src_x]
        return out

    def _get_params(self) -> dict:
        return {"beta": self.beta, "k": self.k, "max_iter": self.max_iter}
