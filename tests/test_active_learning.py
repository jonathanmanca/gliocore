"""
tests/test_active_learning.py — verifica che il loop di apprendimento attivo
del Random Forest sia realmente collegato:

  • _load_training_data legge i campioni accumulati nello store
  • i campioni di una modalità diversa (feature dim ≠) vengono ignorati
  • con abbastanza correzioni l'RF si addestra su quelle, non sul bootstrap GMM

Eseguibile con `pytest` oppure direttamente: `python tests/test_active_learning.py`.
"""
from __future__ import annotations
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation import bayesian_rf
from segmentation.bayesian_rf import BayesianRFSegmentation
from io_data.modality import build_feature_set, Modality, SegmentationContext


def _synthetic_pet():
    """Crea un piccolo caso PET sintetico (2 canali, maschera interna)."""
    shape = (12, 12, 12)
    rng = np.random.RandomState(0)
    mask = np.zeros(shape, dtype=bool)
    mask[2:10, 2:10, 2:10] = True            # 512 voxel
    vols = {
        "suvr": rng.rand(*shape).astype("float32"),
        "suv":  rng.rand(*shape).astype("float32"),
    }
    fs = build_feature_set(vols, mask, Modality.PET, normalize=True)
    ctx = SegmentationContext(modality=Modality.PET, full_volumes=vols,
                              primary_channel="suvr")
    return fs, ctx, mask


def _isolate_store(tmp: Path, monkey=None):
    """Reindirizza store e percorsi modello in una cartella temporanea."""
    bayesian_rf._TRAIN_DIR = tmp / "active_learning"
    bayesian_rf._MODEL_PATH = tmp / "rf_model.pkl"
    bayesian_rf._SCALER_PATH = tmp / "rf_scaler.pkl"


def test_empty_store_returns_none():
    with tempfile.TemporaryDirectory() as td:
        _isolate_store(Path(td))
        rf = BayesianRFSegmentation()
        assert rf._load_training_data(None, n_features=5) is None


def test_loop_trains_on_corrections_and_filters_modality():
    with tempfile.TemporaryDirectory() as td:
        _isolate_store(Path(td))
        fs, ctx, mask = _synthetic_pet()
        rf = BayesianRFSegmentation()

        # Feature aumentate come le produce l'RF (PET: 2 canali + 3 statistiche = 5)
        X = rf._augment_features(fs, ctx)
        n_feat = X.shape[1]
        assert n_feat == 5, f"atteso 5 feature, ottenute {n_feat}"

        y = np.random.RandomState(1).randint(0, 3, size=len(X))

        # 1) campione coerente (PET) + 2) campione di un'altra modalità (4 feature)
        BayesianRFSegmentation.save_training_sample(X, y, "PAZ_PET", tag="GMM")
        wrong = np.random.RandomState(2).rand(200, 4).astype("float32")
        BayesianRFSegmentation.save_training_sample(
            wrong, np.zeros(200, dtype=int), "PAZ_MRI", tag="GMM")

        # _load_training_data deve tenere solo il campione coerente
        loaded = rf._load_training_data(ctx, n_features=n_feat)
        assert loaded is not None
        Xl, yl = loaded
        assert len(yl) == len(y), "il campione di altra modalità non è stato filtrato"
        assert Xl.shape[1] == n_feat

        # fit completo → deve addestrarsi sulle correzioni, non sul bootstrap GMM
        result = rf.fit(fs, ctx)
        assert result.metrics["training_source"] == "corrections"
        assert result.metrics["n_training"] == len(y)
        assert bayesian_rf._MODEL_PATH.exists(), "il modello addestrato va persistito"


def _run_all():
    test_empty_store_returns_none()
    print("✔ test_empty_store_returns_none")
    test_loop_trains_on_corrections_and_filters_modality()
    print("✔ test_loop_trains_on_corrections_and_filters_modality")
    print("\nTutti i test passati.")


if __name__ == "__main__":
    _run_all()
