"""
segmentation/gmm.py — Gaussian Mixture Model multivariato.

Ora accetta feature matrix N×D (non più solo SUVR scalare).
- PET:      tipicamente [SUVR] o [SUVR, SUV]
- MRI:      [T1, T1ce, T2, FLAIR]
- PET_MRI:  tutti i canali disponibili

La selezione di k usa BIC/AIC sulla feature matrix completa.
L'ordering dei cluster avviene nella base class sulla feature primaria.
"""
from __future__ import annotations
import logging
import warnings
from typing import Literal

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext

log = logging.getLogger(__name__)


class GMMSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 6,
        criterion: Literal["BIC", "AIC"] = "BIC",
        n_init: int = 10,
        covariance_type: str = "full",
        min_voxels: int = 50,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.criterion = criterion
        self.n_init = n_init
        self.covariance_type = covariance_type
        self.min_voxels = min_voxels

    @property
    def name(self) -> str:
        return "GMM"

    @property
    def description(self) -> str:
        return (
            "Multivariate Gaussian Mixture Model with automatic k selection "
            f"({self.criterion}). Works on single-feature or multi-channel data. "
            "Clusters are ordered by the primary feature of the modality."
        )

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        X = features.matrix
        n = len(X)
        if n < self.min_voxels:
            raise ValueError(f"GMM: too few voxels ({n}, minimum {self.min_voxels})")

        k_max_safe = min(self.k_max, n // 10)
        model_results = []
        best_score, best_gmm, best_k = np.inf, None, self.k_min

        for k in range(self.k_min, k_max_safe + 1):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=self.covariance_type,
                    n_init=self.n_init,
                    random_state=42,
                )
                gmm.fit(X)
                converged = not any(
                    issubclass(x.category, ConvergenceWarning) for x in w
                )
            bic = gmm.bic(X)
            aic = gmm.aic(X)
            score = bic if self.criterion == "BIC" else aic
            model_results.append({"k": k, "BIC": bic, "AIC": aic, "converged": converged})
            if score < best_score:
                best_score, best_gmm, best_k = score, gmm, k

        log.info(f"GMM [{context.modality.value}]: best k={best_k} "
                 f"({self.criterion}={best_score:.1f}), {features.n_features} features")

        labels     = best_gmm.predict(X)             # 0-indexed
        membership = best_gmm.predict_proba(X)        # (N, k)

        extra = {
            "best_k":     best_k,
            "BIC":        best_gmm.bic(X),
            "AIC":        best_gmm.aic(X),
            "criterion":  self.criterion,
            "all_models": model_results,
        }
        return labels, membership, extra

    def _get_params(self) -> dict:
        return {
            "k_min": self.k_min, "k_max": self.k_max,
            "criterion": self.criterion, "n_init": self.n_init,
            "covariance_type": self.covariance_type,
        }
