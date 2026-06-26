"""
segmentation/level_set.py — Chan-Vese Level Set.

Modello intensità-based: opera su UN canale (la feature primaria della modalità).
Per natura il level set lavora su un campo scalare, quindi sceglie il canale
più informativo: SUVR per PET, T1ce per MRI.

Dichiara esplicitamente che è meno adatto a feature multivariate:
usa solo il canale primario, ignorando gli altri.
"""
from __future__ import annotations
import logging
import numpy as np
from skimage.segmentation import morphological_chan_vese, chan_vese

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext, Modality

log = logging.getLogger(__name__)


class LevelSetSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        num_iter: int = 200,
        smoothing: int = 3,
        method: str = "morphological",
        n_clusters: int = 2,
    ):
        self.num_iter = num_iter
        self.smoothing = smoothing
        self.method = method
        self.n_clusters = n_clusters

    @property
    def name(self) -> str:
        return "LevelSet"

    @property
    def description(self) -> str:
        return (
            "Chan-Vese Level Set on the primary channel (SUVR for PET, T1ce for MRI). "
            "Intensity-based model: uses a single feature, not multivariate. "
            "Refines the contour and splits the interior by threshold."
        )

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        # Level Set ha bisogno del volume 3D completo del canale primario
        primary_vol = context.primary_volume
        if primary_vol is None:
            raise ValueError(
                f"LevelSet requires the full volume of channel "
                f"'{context.primary_channel}', not available in the context."
            )

        mask = features.mask
        refined = self._run_slices(primary_vol, mask)
        if refined.sum() == 0:
            log.warning("LevelSet: empty contour, using original mask")
            refined = mask.copy()

        # Suddivisione interna per soglia sul canale primario
        primary_1d = features.primary_feature
        refined_vals = primary_vol[refined]

        if self.n_clusters == 2:
            thr = float(np.median(primary_1d))
            labels_refined = np.where(refined_vals < thr, 0, 1)
        else:
            q33, q66 = np.percentile(primary_1d, [33, 66])
            labels_refined = np.zeros(len(refined_vals), dtype=np.int32)
            labels_refined[refined_vals >= q33] = 1
            labels_refined[refined_vals >= q66] = 2

        # Costruisci labels nello spazio dei voxel mascherati originali
        # I voxel raffinati prendono la loro etichetta, gli altri restano fuori
        full_labels = np.full(features.n_voxels, -1, dtype=np.int32)
        refined_in_mask = refined[mask]   # bool sui voxel mascherati
        full_labels[refined_in_mask] = labels_refined

        extra = {
            "best_k": self.n_clusters,
            "method": self.method,
            "refined_voxels": int(refined.sum()),
            "note": "Level Set uses only the primary channel (intensity-based)",
        }
        return full_labels, None, extra

    def _run_slices(self, volume, mask):
        refined = np.zeros_like(mask, dtype=bool)
        z_idx = np.where(mask.any(axis=(0, 1)))[0]
        for z in z_idx:
            sl = volume[:, :, z]
            sm = mask[:, :, z]
            if not sm.any():
                continue
            vmin, vmax = sl.min(), sl.max()
            if vmax == vmin:
                continue
            norm = (sl - vmin) / (vmax - vmin)
            try:
                if self.method == "morphological":
                    res = morphological_chan_vese(
                        norm, num_iter=self.num_iter, smoothing=self.smoothing,
                        init_level_set=sm.astype(float),
                    )
                else:
                    res = chan_vese(
                        norm, mu=0.25, max_num_iter=self.num_iter,
                        init_level_set=sm.astype(float),
                    )
                refined[:, :, z] = res.astype(bool) & sm
            except Exception as e:
                log.warning(f"LevelSet slice z={z}: {e}")
                refined[:, :, z] = sm
        return refined

    def _get_params(self) -> dict:
        return {"num_iter": self.num_iter, "n_clusters": self.n_clusters, "method": self.method}
