"""
segmentation/registry.py — Registro modelli modality-aware.

Ogni modello dichiara quali modalità supporta e se richiede il volume completo.
"""
from __future__ import annotations
from typing import Type

from .base import BaseSegmentationModel
from .gmm import GMMSegmentation
from .fuzzy_cmeans import FuzzyCMeansSegmentation
from .level_set import LevelSetSegmentation
from .mrf_em import MRFEMSegmentation
from .dbscan import DBSCANSegmentation
from .hierarchical import HierarchicalSegmentation
from .threshold_baseline import ThresholdBaseline
from io_data.modality import Modality

_REGISTRY: dict[str, Type[BaseSegmentationModel]] = {
    "GMM":          GMMSegmentation,
    "FCM":          FuzzyCMeansSegmentation,
    "LevelSet":     LevelSetSegmentation,
    "MRF-EM":       MRFEMSegmentation,
    "DBSCAN":       DBSCANSegmentation,
    "Hierarchical": HierarchicalSegmentation,
    "Threshold":    ThresholdBaseline,
}

# BayesianRF è opzionale (richiede pyradiomics) — caricato se disponibile
try:
    from .bayesian_rf import BayesianRFSegmentation
    _REGISTRY["BayesianRF"] = BayesianRFSegmentation
except ImportError:
    pass


def get_model(name: str, **kwargs) -> BaseSegmentationModel:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown model: '{name}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_models() -> list[dict]:
    out = []
    for name, cls in _REGISTRY.items():
        inst = cls()
        out.append({
            "name": name,
            "description": inst.description,
            "modalities": [m.value for m in inst.supported_modalities],
        })
    return out


def list_models_for_modality(modality: Modality) -> list[str]:
    """Restituisce i nomi dei modelli che supportano una modalità."""
    out = []
    for name, cls in _REGISTRY.items():
        if modality in cls().supported_modalities:
            out.append(name)
    return out
