"""
segmentation/dbscan.py — DBSCAN adattivo multivariato.

Usa le feature della modalità + coordinate spaziali normalizzate.
ε stimato dal knee-point del k-distance graph.
"""
from __future__ import annotations
import logging
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext

log = logging.getLogger(__name__)


class DBSCANSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        eps: float | None = None,
        min_samples: int | None = None,
        spatial_weight: float = 0.3,
        n_neighbors_eps: int = 5,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.spatial_weight = spatial_weight
        self.n_neighbors_eps = n_neighbors_eps

    @property
    def name(self) -> str:
        return "DBSCAN"

    @property
    def description(self) -> str:
        return (
            "Adaptive DBSCAN on modality features + spatial coordinates. "
            "Does not require k, finds clusters of arbitrary shape. "
            "Outlier voxels (label 0) = transition zones."
        )

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        n = features.n_voxels
        if n < 20:
            raise ValueError(f"DBSCAN: too few voxels ({n})")

        # Feature già normalizzate + coordinate spaziali
        coords = np.column_stack(np.where(features.mask)).astype(np.float32)
        coords_norm = coords / np.array(features.shape, dtype=np.float32)

        X = np.hstack([
            features.matrix,                        # feature di modalità (già norm)
            coords_norm * self.spatial_weight,      # coordinate pesate
        ])

        eps = self.eps or self._estimate_eps(X, self.n_neighbors_eps)
        min_samples = self.min_samples or max(5, int(n * 0.01))
        log.info(f"DBSCAN [{context.modality.value}]: ε={eps:.4f}, min_samples={min_samples}")

        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X)   # -1 = noise

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())

        # Fallback: se nessun cluster, allarga ε progressivamente
        retry = 0
        while n_clusters == 0 and retry < 3:
            eps *= 1.5
            retry += 1
            log.warning(f"DBSCAN: no cluster, retrying with ε={eps:.4f}")
            labels = DBSCAN(eps=eps, min_samples=max(3, min_samples // 2),
                            n_jobs=-1).fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int((labels == -1).sum())

        if n_clusters == 0:
            raise ValueError(
                "DBSCAN found no clusters even after retry. "
                "The data may be too uniform or too noisy."
            )

        log.info(f"DBSCAN: {n_clusters} clusters, {n_noise} outliers ({n_noise/n:.1%})")

        extra = {
            "best_k":          n_clusters,
            "eps":             round(float(eps), 4),
            "min_samples":     min_samples,
            "n_noise_voxels":  n_noise,
            "noise_fraction":  round(n_noise / n, 4),
        }
        # labels con -1 vengono gestiti dall'ordering (diventano 0)
        return labels, None, extra

    @staticmethod
    def _estimate_eps(X, k) -> float:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        kd = np.sort(distances[:, -1])
        d2 = np.diff(np.diff(kd))
        if len(d2) == 0:
            return float(np.percentile(kd, 90))
        knee = int(np.argmax(d2)) + 2
        eps = float(kd[min(knee, len(kd) - 1)])
        p10, p90 = np.percentile(kd, [10, 90])
        if eps < p10 or eps > p90:
            eps = float(np.percentile(kd, 50))
        return eps

    def _get_params(self) -> dict:
        return {"spatial_weight": self.spatial_weight, "eps": self.eps}
