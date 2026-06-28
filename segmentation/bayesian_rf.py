"""
segmentation/bayesian_rf.py — Random Forest con apprendimento attivo (multi-modale).

Aggiornato per la nuova firma fit(features, context).
Supervisionato: si addestra sulle correzioni manuali accumulate.
Feature: usa la matrice multivariata della modalità + statistiche locali.

Riferimento: van der Voort et al. (2023) Neuro-Oncology
             DOI: 10.1093/neuonc/noac166
"""
from __future__ import annotations
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext, Modality

log = logging.getLogger(__name__)

_MODEL_PATH  = Path(__file__).parent.parent / "data" / "rf_model.pkl"
_SCALER_PATH = Path(__file__).parent.parent / "data" / "rf_scaler.pkl"
# Store dell'apprendimento attivo: una sottocartella per modello, file .npz
# con le coppie (feature aumentate, label corretti) salvate a ogni correzione.
_TRAIN_DIR   = Path(__file__).parent.parent / "data" / "active_learning"
MIN_TRAINING_SAMPLES = 100


class BayesianRFSegmentation(BaseSegmentationModel):

    def __init__(
        self,
        n_estimators: int = 200,
        min_samples: int = MIN_TRAINING_SAMPLES,
        k_bootstrap: int = 3,
    ):
        self.n_estimators = n_estimators
        self.min_samples = min_samples
        self.k_bootstrap = k_bootstrap
        self._rf = None
        self._scaler = None
        self._load_model()

    @property
    def name(self) -> str:
        return "BayesianRF"

    @property
    def description(self) -> str:
        status = "trained" if self._rf is not None else "not trained"
        return (
            "Random Forest with active learning. It trains on manual "
            f"corrections (status: {status}). Multivariate: uses all the features of "
            "the modality + local statistics. First run: bootstrap from GMM."
        )

    @property
    def supported_modalities(self) -> set:
        # RF funziona su tutte le modalità (feature generiche)
        return {Modality.PET, Modality.MRI, Modality.PET_MRI}

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        X = self._augment_features(features, context)

        # Training: correzioni reali accumulate o, in mancanza, bootstrap da GMM.
        # Le correzioni sono filtrate per dimensionalità coerente con X (le
        # modalità con un numero diverso di canali non vanno mescolate).
        training = self._load_training_data(context, X.shape[1])
        trained_on_corrections = bool(training and len(training[1]) >= self.min_samples)
        if trained_on_corrections:
            X_train, y_train = training
            log.info(f"BayesianRF: training on {len(y_train)} real correction samples")
        else:
            log.info("BayesianRF: synthetic bootstrap from GMM")
            X_train, y_train = self._bootstrap_gmm(X, features)

        self._train(X_train, y_train)
        if trained_on_corrections:
            # Persisti il modello addestrato sulle correzioni per riusi futuri.
            try:
                self._save_model()
            except Exception as e:
                log.warning(f"BayesianRF: model not saved ({e})")

        proba = self._rf.predict_proba(self._scaler.transform(X))
        labels = np.argmax(proba, axis=1)   # 0-indexed

        entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
        extra = {
            "best_k": len(np.unique(labels)),
            "n_training": len(y_train),
            "training_source": "corrections" if trained_on_corrections else "gmm_bootstrap",
            "mean_uncertainty": round(float(entropy.mean()), 4),
            "high_uncertainty_pct": round(float((entropy > 0.5).mean() * 100), 2),
        }
        return labels, proba, extra

    def _augment_features(self, features: FeatureSet, context: SegmentationContext):
        """Aggiunge statistiche locali alle feature di modalità."""
        from scipy.ndimage import uniform_filter, generic_gradient_magnitude, sobel

        X = features.matrix.copy()
        primary_vol = context.primary_volume
        if primary_vol is not None:
            mask = features.mask
            local_mean = uniform_filter(primary_vol, size=3)[mask]
            sq_mean = uniform_filter(primary_vol**2, size=3)[mask]
            local_std = np.sqrt(np.maximum(sq_mean - local_mean**2, 0))
            grad = generic_gradient_magnitude(primary_vol, sobel)[mask]
            X = np.column_stack([X, local_mean, local_std, grad])
        return X.astype(np.float32)

    def _bootstrap_gmm(self, X_aug, features):
        from sklearn.mixture import GaussianMixture
        X = features.matrix
        best_bic, best_gmm = np.inf, None
        for k in range(2, min(6, len(X)//20 + 1)):
            gmm = GaussianMixture(n_components=k, n_init=3, random_state=42)
            gmm.fit(X)
            if gmm.bic(X) < best_bic:
                best_bic, best_gmm = gmm.bic(X), gmm
        labels = best_gmm.predict(X)
        return X_aug, labels

    def _train(self, X, y):
        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._rf = RandomForestClassifier(
            n_estimators=self.n_estimators, min_samples_leaf=5,
            class_weight="balanced", n_jobs=-1, random_state=42,
        )
        self._rf.fit(Xs, y)

    def retrain_from_corrections(self, corrections: list[dict]) -> dict:
        if not corrections:
            return {"status": "no corrections"}
        feats, labs = [], []
        for c in corrections:
            if "features" in c and "labels" in c:
                feats.append(np.array(c["features"]))
                labs.append(np.array(c["labels"]))
        if not feats:
            return {"status": "no valid data"}
        X = np.vstack(feats); y = np.concatenate(labs)
        self._train(X, y)
        self._save_model()
        return {"status": "retrained", "n_samples": len(y)}

    @classmethod
    def training_store_dir(cls, model_name: str = "BayesianRF") -> Path:
        """Cartella dei campioni di apprendimento attivo per un modello."""
        return _TRAIN_DIR / model_name

    @classmethod
    def save_training_sample(
        cls,
        X: np.ndarray,
        y: np.ndarray,
        patient_id: str,
        tag: str = "",
    ) -> Path:
        """
        Persiste una coppia (feature aumentate, label corretti) nello store.

        Chiamata dall'UI quando l'utente salva una correzione: X deve essere
        prodotta da `_augment_features` sulla stessa modalità, y i label corretti
        allineati alle righe di X (cioè i voxel della maschera).
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(np.int64).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(
                f"save_training_sample: X{X.shape} e y{y.shape} non allineati"
            )
        d = cls.training_store_dir()
        d.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_tag = (tag or "edit").replace("/", "_")
        path = d / f"{patient_id}_{safe_tag}_{ts}.npz"
        np.savez_compressed(path, X=X, y=y)
        log.info(f"Active learning: saved {len(y)} samples → {path.name}")
        return path

    def _load_training_data(self, context, n_features: int):
        """
        Carica tutte le correzioni accumulate compatibili con la modalità
        corrente (stesso numero di feature di X) e le concatena in (X, y).
        Restituisce None se non ci sono campioni utilizzabili.
        """
        d = self.training_store_dir(self.name)
        if not d.exists():
            return None
        Xs, ys, skipped = [], [], 0
        for f in sorted(d.glob("*.npz")):
            try:
                arr = np.load(f)
                X_f, y_f = arr["X"], arr["y"]
            except Exception as e:
                log.warning(f"Active learning: skip {f.name} ({e})")
                continue
            if X_f.ndim != 2 or X_f.shape[1] != n_features:
                skipped += 1   # campione di un'altra modalità/feature set
                continue
            Xs.append(X_f)
            ys.append(y_f)
        if not Xs:
            if skipped:
                log.info(f"Active learning: {skipped} samples skipped "
                         f"(feature dim ≠ {n_features})")
            return None
        return np.vstack(Xs), np.concatenate(ys)

    def _save_model(self):
        _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_PATH, "wb") as f:
            pickle.dump(self._rf, f)
        with open(_SCALER_PATH, "wb") as f:
            pickle.dump(self._scaler, f)

    def _load_model(self):
        if _MODEL_PATH.exists() and _SCALER_PATH.exists():
            try:
                with open(_MODEL_PATH, "rb") as f:
                    self._rf = pickle.load(f)
                with open(_SCALER_PATH, "rb") as f:
                    self._scaler = pickle.load(f)
            except Exception as e:
                log.warning(f"BayesianRF: loading failed ({e})")

    def _get_params(self) -> dict:
        return {"n_estimators": self.n_estimators}
