"""
segmentation/base.py — Interfaccia comune modality-aware.

Nuova firma pulita:
    result = model.fit(features: FeatureSet, context: SegmentationContext)

I modelli ricevono una matrice di feature N×D (non più solo SUVR scalare)
e un contesto con i volumi completi e la modalità.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from io_data.modality import (
    FeatureSet, SegmentationContext, Modality,
    order_clusters_by_feature,
)


@dataclass
class SegmentationResult:
    """Output standardizzato di qualsiasi modello."""
    label_volume: np.ndarray              # 3D, 0=fuori, 1..k=cluster
    n_clusters:   int
    hypo_mask:    np.ndarray              # 3D bool — cluster con feature primaria minima
    hyper_mask:   np.ndarray              # 3D bool — cluster con feature primaria massima
    metrics:      dict[str, Any]
    model_name:   str
    modality:     str                     # modalità usata
    params:       dict[str, Any]
    membership:   np.ndarray | None = None
    primary_channel: str = "suvr"         # canale usato per l'ordering

    def cluster_mask(self, label: int) -> np.ndarray:
        return self.label_volume == label


class BaseSegmentationModel(ABC):
    """
    Classe base per tutti i modelli di segmentazione.

    Sottoclassi devono implementare:
      - name, description (proprietà)
      - supported_modalities (quali modalità il modello gestisce bene)
      - _fit_impl(features, context) → (labels_1d, membership, extra_metrics)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def supported_modalities(self) -> set[Modality]:
        """Modalità supportate. Default: tutte."""
        return {Modality.PET, Modality.MRI, Modality.PET_MRI}

    @abstractmethod
    def _fit_impl(
        self,
        features: FeatureSet,
        context:  SegmentationContext,
    ) -> tuple[np.ndarray, np.ndarray | None, dict]:
        """
        Implementazione specifica del modello.

        Returns:
          labels    : (N,) etichette cluster (0-indexed o con -1 per outlier)
          membership: (N, k) probabilità o None
          metrics   : dict di metriche interne
        """

    def fit(
        self,
        features: FeatureSet,
        context:  SegmentationContext,
    ) -> SegmentationResult:
        """
        Esegue la segmentazione completa: fit + ordering + volumi.

        Questo metodo è condiviso da tutti i modelli e gestisce:
        - controllo modalità supportata
        - chiamata a _fit_impl
        - ordering dei cluster per feature primaria
        - costruzione dei volumi 3D
        """
        if context.modality not in self.supported_modalities:
            raise ValueError(
                f"{self.name} does not support modality {context.modality.value}. "
                f"Supported: {[m.value for m in self.supported_modalities]}"
            )

        # Fit specifico del modello
        labels, membership, extra = self._fit_impl(features, context)

        # Ordering dei cluster per feature primaria
        primary_1d = features.primary_feature
        ordered_labels, sort_idx = order_clusters_by_feature(
            labels, primary_1d, order=context.cluster_order,
        )

        n_clusters = int(len([c for c in np.unique(ordered_labels) if c > 0]))

        # Riordina la membership se presente
        membership_ordered = None
        if membership is not None and len(sort_idx) > 0:
            valid_idx = [int(c) for c in sort_idx if c < membership.shape[1]]
            if len(valid_idx) == membership.shape[1]:
                membership_ordered = membership[:, valid_idx]
            else:
                membership_ordered = membership

        # Costruisci volumi 3D
        mask = features.mask
        label_vol  = self._build_label_volume(mask, ordered_labels)
        hypo_mask  = self._build_binary(mask, ordered_labels == 1)
        hyper_mask = self._build_binary(mask, ordered_labels == n_clusters)

        # Metriche base + extra del modello
        metrics = {
            "n_clusters":      n_clusters,
            "n_voxels":        features.n_voxels,
            "n_features":      features.n_features,
            "channels":        features.channel_names,
            "primary_channel": context.primary_channel,
            "modality":        context.modality.value,
            **extra,
        }

        return SegmentationResult(
            label_volume=label_vol,
            n_clusters=n_clusters,
            hypo_mask=hypo_mask,
            hyper_mask=hyper_mask,
            metrics=metrics,
            model_name=self.name,
            modality=context.modality.value,
            params=self._get_params(),
            membership=membership_ordered,
            primary_channel=context.primary_channel,
        )

    def _get_params(self) -> dict:
        """Override per esporre i parametri del modello."""
        return {}

    @staticmethod
    def _build_label_volume(mask: np.ndarray, labels: np.ndarray) -> np.ndarray:
        vol = np.zeros(mask.shape, dtype=np.uint8)
        vol[mask] = labels
        return vol

    @staticmethod
    def _build_binary(mask: np.ndarray, condition: np.ndarray) -> np.ndarray:
        vol = np.zeros(mask.shape, dtype=bool)
        vol[mask] = condition
        return vol
