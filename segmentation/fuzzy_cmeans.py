"""
segmentation/fuzzy_cmeans.py — Fuzzy C-Means multivariato.

Accetta feature matrix N×D. skfuzzy lavora su (n_features, n_samples)
quindi la matrice viene trasposta internamente.
"""
from __future__ import annotations
import logging
import numpy as np

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext

log = logging.getLogger(__name__)


class FuzzyCMeansSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        k: int = 3,
        m: float = 2.0,
        error: float = 1e-5,
        maxiter: int = 300,
        auto_k: bool = False,
        k_min: int = 2,
        k_max: int = 6,
    ):
        self.k = k
        self.m = m
        self.error = error
        self.maxiter = maxiter
        self.auto_k = auto_k
        self.k_min = k_min
        self.k_max = k_max

    @property
    def name(self) -> str:
        return "FCM"

    @property
    def description(self) -> str:
        return (
            "Multivariate Fuzzy C-Means: each voxel has continuous membership [0,1] "
            "to every cluster. Works on single-feature or multi-channel data. "
            "Optional k selection via Xie-Beni index."
        )

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        try:
            import skfuzzy as fuzz
        except ImportError:
            raise ImportError("Install scikit-fuzzy: pip install scikit-fuzzy")

        # skfuzzy: (n_features, n_samples)
        X = features.matrix.T

        best_k = self.k
        if self.auto_k:
            best_k = self._select_k(X, fuzz)
            log.info(f"FCM auto-k: k={best_k}")

        centers, U, _, _, _, _, fpc = fuzz.cluster.cmeans(
            X, c=best_k, m=self.m, error=self.error,
            maxiter=self.maxiter, init=None, seed=42,
        )

        labels     = np.argmax(U, axis=0)   # 0-indexed
        membership = U.T                     # (N, k)

        extra = {
            "best_k": best_k,
            "FPC":    float(fpc),
        }
        return labels, membership, extra

    def _select_k(self, X, fuzz) -> int:
        """Seleziona k minimizzando Xie-Beni index."""
        best_k, best_xb = self.k_min, np.inf
        for k in range(self.k_min, self.k_max + 1):
            centers, U, *_ = fuzz.cluster.cmeans(
                X, c=k, m=self.m, error=self.error, maxiter=self.maxiter, seed=42
            )
            xb = self._xie_beni(X.T, U, centers)
            if xb < best_xb:
                best_xb, best_k = xb, k
        return best_k

    @staticmethod
    def _xie_beni(X, U, centers) -> float:
        k, N = U.shape
        m = 2.0
        compactness = 0.0
        for i in range(k):
            diff = X - centers[i]
            compactness += np.sum((U[i] ** m) * np.sum(diff ** 2, axis=1))
        compactness /= N
        min_sep = np.inf
        for i in range(k):
            for j in range(k):
                if i != j:
                    d = np.sum((centers[i] - centers[j]) ** 2)
                    min_sep = min(min_sep, d)
        return compactness / min_sep if min_sep > 0 else np.inf

    def _get_params(self) -> dict:
        return {"k": self.k, "m": self.m, "auto_k": self.auto_k}
