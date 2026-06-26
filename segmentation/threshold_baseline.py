"""
segmentation/threshold_baseline.py — Baseline a soglia (riferimento minimo).

Metodo deliberatamente semplice: suddivide i voxel tumorali in k livelli
per quantili della feature primaria. Serve come riferimento inferiore (floor)
nel benchmark: qualunque metodo sofisticato dovrebbe superarlo.

Non ha pretese cliniche: è il controllo che dimostra quanto valore aggiungono
i metodi di clustering rispetto a una semplice sogliatura per intensità.
"""
from __future__ import annotations
import numpy as np

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext, Modality


class ThresholdBaseline(BaseSegmentationModel):

    def __init__(self, k: int = 3):
        self.k = k

    @property
    def name(self) -> str:
        return "Threshold"

    @property
    def description(self) -> str:
        return ("Threshold baseline: splits voxels into k levels by quantiles "
                "of the primary feature. Minimal reference for comparison.")

    @property
    def supported_modalities(self) -> set:
        return {Modality.PET, Modality.MRI, Modality.PET_MRI}

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        primary = features.primary_feature
        # Quantili per dividere in k fasce di intensità
        edges = np.quantile(primary, np.linspace(0, 1, self.k + 1)[1:-1])
        labels = np.digitize(primary, edges)  # 0..k-1
        extra = {"best_k": self.k, "method": "quantile threshold"}
        return labels, None, extra

    def _get_params(self) -> dict:
        return {"k": self.k}
