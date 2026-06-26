"""
segmentation/validation.py — Suite validazione k modality-aware.

I pesi dei criteri cambiano per modalità:

PET (feature scalare SUVR):
  - Silhouette e Davies-Bouldin sono affidabili
  - BIC sovrastima k → peso basso
  - Calinski-Harabasz buono

MRI (feature 4D T1/T1ce/T2/FLAIR):
  - In alta dimensione Silhouette diventa meno discriminante (curse of dimensionality)
    ma resta utile → peso ridotto rispetto a PET
  - Calinski-Harabasz e Davies-Bouldin più robusti in multi-D → peso maggiore
  - BIC su feature multivariata è più sensato che su scalare → peso medio

PET_MRI (feature mista):
  - Compromesso tra i due
  - Davies-Bouldin e Calinski-Harabasz pesati di più per robustezza multi-D

Nota onesta: nessuna metrica è perfetta. Il voto pesato + ispezione visiva
resta la strategia migliore. ARI NON è incluso qui perché richiede ground truth
(non disponibile in fase di selezione k); è usato solo nel benchmark BraTS.
"""
from __future__ import annotations
import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
)

from io_data.modality import Modality

log = logging.getLogger(__name__)


# Pesi dei criteri per modalità — scelti per significatività reale, non per completezza
WEIGHTS_BY_MODALITY = {
    Modality.PET: {
        "silhouette": 0.30, "calinski_harabasz": 0.25,
        "davies_bouldin": 0.25, "bic": 0.10, "elbow": 0.10,
    },
    Modality.MRI: {
        # In 4D Silhouette perde potere → ridotta; CH/DB più affidabili
        "silhouette": 0.15, "calinski_harabasz": 0.30,
        "davies_bouldin": 0.30, "bic": 0.15, "elbow": 0.10,
    },
    Modality.PET_MRI: {
        "silhouette": 0.20, "calinski_harabasz": 0.28,
        "davies_bouldin": 0.28, "bic": 0.14, "elbow": 0.10,
    },
}


@dataclass
class KResult:
    k: int
    bic: float
    aic: float
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    inertia: float
    stability: float = 0.0
    labels: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ValidationReport:
    modality: str
    results: list[KResult]
    k_recommended: int
    k_per_criterion: dict
    consensus: bool
    explanation: str

    def summary(self) -> str:
        lines = [
            f"━━ k validation [{self.modality}] ━━",
            f"Recommended k: {self.k_recommended}",
            f"Consensus: {'✔' if self.consensus else '⚠ partial'}",
            "",
            "Votes per criterion:",
        ]
        for crit, k in self.k_per_criterion.items():
            lines.append(f"  {crit}: k={k}")
        lines += ["", "Explanation:", f"  {self.explanation}"]
        return "\n".join(lines)


class ClusterValidator:
    """Validazione k che adatta i pesi alla modalità."""

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 6,
        n_init: int = 10,
        n_bootstrap: int = 10,
        covariance_type: str = "full",
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.n_init = n_init
        self.n_bootstrap = n_bootstrap
        self.covariance_type = covariance_type

    def validate(self, X: np.ndarray, modality: Modality) -> ValidationReport:
        """
        X : feature matrix (N, D) — già normalizzata
        modality : determina i pesi dei criteri
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(X)
        k_max_safe = min(self.k_max, n // 20)
        if k_max_safe < self.k_min:
            raise ValueError(f"Too few voxels ({n}) for k≥{self.k_min}")

        weights = WEIGHTS_BY_MODALITY[modality]
        log.info(f"k validation [{modality.value}]: k={self.k_min}..{k_max_safe}, "
                 f"D={X.shape[1]}, adapted weights")

        results = [self._eval_k(X, k) for k in range(self.k_min, k_max_safe + 1)]
        if self.n_bootstrap > 0:
            self._add_stability(X, results)

        k_per = {
            "BIC":              results[np.argmin([r.bic for r in results])].k,
            "Silhouette":       results[np.argmax([r.silhouette for r in results])].k,
            "Calinski-Harabasz": results[np.argmax([r.calinski_harabasz for r in results])].k,
            "Davies-Bouldin":   results[np.argmin([r.davies_bouldin for r in results])].k,
            "Elbow":            self._elbow([r.inertia for r in results], [r.k for r in results]),
        }
        k_vote = self._weighted_vote(results, weights)
        k_per["Weighted vote"] = k_vote

        votes = list(k_per.values())
        counts = {k: votes.count(k) for k in set(votes)}
        k_rec = max(counts, key=counts.get)
        consensus = counts[k_rec] >= 3

        explanation = self._explain(modality, k_rec, k_per, counts, results, X.shape[1])

        return ValidationReport(
            modality=modality.value, results=results, k_recommended=k_rec,
            k_per_criterion=k_per, consensus=consensus, explanation=explanation,
        )

    def _eval_k(self, X, k) -> KResult:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gmm = GaussianMixture(n_components=k, covariance_type=self.covariance_type,
                                  n_init=self.n_init, random_state=42)
            gmm.fit(X)
        labels = gmm.predict(X)
        inertia = sum(float(np.sum((X[labels == c] - gmm.means_[c])**2))
                      for c in range(k) if (labels == c).any())
        # Silhouette su sample se troppi voxel
        if len(X) > 5000:
            idx = np.random.default_rng(42).choice(len(X), 5000, replace=False)
            sil = silhouette_score(X[idx], labels[idx])
        else:
            sil = silhouette_score(X, labels)
        return KResult(
            k=k, bic=gmm.bic(X), aic=gmm.aic(X), silhouette=sil,
            calinski_harabasz=calinski_harabasz_score(X, labels),
            davies_bouldin=davies_bouldin_score(X, labels),
            inertia=inertia, labels=labels,
        )

    def _add_stability(self, X, results):
        from sklearn.metrics import adjusted_rand_score
        rng = np.random.default_rng(42)
        n = len(X)
        for r in results:
            aris = []
            for _ in range(self.n_bootstrap):
                idx = rng.choice(n, n, replace=True)
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    g = GaussianMixture(n_components=r.k, covariance_type=self.covariance_type,
                                        n_init=3, random_state=None)
                    g.fit(X[idx])
                aris.append(adjusted_rand_score(r.labels, g.predict(X)))
            r.stability = float(np.mean(aris))

    @staticmethod
    def _elbow(inertias, ks):
        if len(inertias) < 3:
            return ks[0]
        d2 = np.diff(np.diff(np.array(inertias, dtype=float)))
        return ks[max(0, min(int(np.argmax(d2)) + 1, len(ks) - 1))]

    def _weighted_vote(self, results, weights):
        ks = [r.k for r in results]
        scores = np.zeros(len(results))

        def nmax(v):
            v = np.array(v, float); r = v.max() - v.min()
            return (v - v.min()) / r if r > 0 else np.full_like(v, 0.5)
        def nmin(v):
            return 1 - nmax(v)

        scores += weights.get("silhouette", 0) * nmax([r.silhouette for r in results])
        scores += weights.get("calinski_harabasz", 0) * nmax([r.calinski_harabasz for r in results])
        scores += weights.get("davies_bouldin", 0) * nmin([r.davies_bouldin for r in results])
        scores += weights.get("bic", 0) * nmin([r.bic for r in results])
        elbow_k = self._elbow([r.inertia for r in results], ks)
        elbow_s = np.zeros(len(results))
        elbow_s[ks.index(elbow_k)] = 1.0
        scores += weights.get("elbow", 0) * elbow_s
        return ks[int(np.argmax(scores))]

    def _explain(self, modality, k_rec, k_per, counts, results, n_dim):
        parts = []
        if counts[k_rec] >= 4:
            parts.append(f"k={k_rec} chosen by {counts[k_rec]}/6 criteria (strong consensus).")
        elif counts[k_rec] == 3:
            parts.append(f"k={k_rec} chosen by 3/6 criteria (moderate consensus).")
        else:
            parts.append(f"k={k_rec} from weighted vote; criteria diverge, check the plots.")

        if modality == Modality.MRI:
            parts.append(f"In MRI ({n_dim}D) Silhouette is less discriminative: "
                         "reduced weight in favor of Calinski-Harabasz and Davies-Bouldin.")
        elif modality == Modality.PET:
            parts.append("In PET (scalar) Silhouette is reliable; BIC overestimates k "
                         "so it has reduced weight.")
        else:
            parts.append("In PET+MRI balanced weights for multi-dimensional robustness.")
        return " ".join(parts)
