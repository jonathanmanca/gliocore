"""
segmentation/hierarchical.py — Segmentazione gerarchica multi-livello.

Contributo metodologico originale di GlioLab.

Invece di un clustering piatto a k cluster, procede in modo gerarchico
rispecchiando la struttura biologica del tumore:

  Livello 1: separa il tessuto in macro-regioni per la feature primaria
             (es. metabolicamente attivo vs spento, o ipo- vs iper-intenso)
  Livello 2: suddivide ogni macro-regione nei suoi sotto-componenti

Questo approccio:
- Rispecchia la gerarchia anatomica (necrosi ⊂ core, enhancement ⊂ attivo)
- Evita lo sbilanciamento dei cluster tipico del k piatto
- È più robusto quando una sottoregione è poco rappresentata

Funziona su qualsiasi modalità grazie al sistema FeatureSet.

Riferimento concettuale:
  La segmentazione gerarchica di tumori è ispirata alla struttura
  multi-livello delle annotazioni BraTS (Menze et al. 2015, IEEE TMI).
"""
from __future__ import annotations
import logging
import warnings

import numpy as np
from sklearn.mixture import GaussianMixture

from .base import BaseSegmentationModel
from io_data.modality import FeatureSet, SegmentationContext, Modality

log = logging.getLogger(__name__)


class HierarchicalSegmentation(BaseSegmentationModel):
    """
    Segmentazione gerarchica a due livelli.

    Parametri
    ---------
    primary_weight : peso extra dato alla feature primaria al livello 1
                     (>1 enfatizza la separazione metabolica/strutturale principale)
    n_level1       : numero di macro-regioni al primo livello (default 2)
    n_sub          : suddivisioni per macro-regione al secondo livello (default 2)
    covariance_type: tipo di covarianza GMM
    """

    def __init__(
        self,
        primary_weight: float = 1.5,
        n_level1: int = 2,
        n_sub: int = 2,
        split_mode: str = "active_only",
        covariance_type: str = "full",
        n_init: int = 5,
    ):
        self.primary_weight = primary_weight
        self.n_level1 = n_level1
        self.n_sub = n_sub
        self.split_mode = split_mode
        self.covariance_type = covariance_type
        self.n_init = n_init

    @property
    def name(self) -> str:
        return "Hierarchical"

    @property
    def description(self) -> str:
        return (
            "Hierarchical segmentation (original GlioLab method). "
            "Level 1: separates inactive tissue (necrosis) vs active. "
            "Level 2: subdivides only the active area into edema/enhancement. "
            "Produces ~3 clusters mapped to necrosis/edema/enhancement, "
            "mirroring the BraTS biological hierarchy. Fast and robust."
        )

    @property
    def supported_modalities(self) -> set:
        return {Modality.PET, Modality.MRI, Modality.PET_MRI}

    def _fit_impl(self, features: FeatureSet, context: SegmentationContext):
        X = features.matrix.copy()
        n = len(X)
        primary_idx = features.primary_idx

        # ── Livello 1: macro-regioni ─────────────────────────────────────────
        # Enfatizza la feature primaria pesandola di più
        X_l1 = X.copy()
        X_l1[:, primary_idx] *= self.primary_weight

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            gmm1 = GaussianMixture(
                n_components=self.n_level1, covariance_type=self.covariance_type,
                n_init=self.n_init, random_state=42,
            )
            gmm1.fit(X_l1)
        macro_labels = gmm1.predict(X_l1)

        # Ordina le macro-regioni per media della feature primaria
        macro_means = {m: X[macro_labels == m, primary_idx].mean()
                       for m in np.unique(macro_labels)}
        macro_order = sorted(macro_means, key=macro_means.get)

        # ── Livello 2: suddivisione ──────────────────────────────────────────
        # In modalità "active_only": divide solo le macro-regioni ATTIVE
        # (alta feature primaria), lascia intatta la necrosi (bassa intensità).
        # Questo produce ~3 cluster: necrosi + (edema, enhancement).
        final_labels = np.zeros(n, dtype=np.int32)
        next_label = 0
        sub_info = {}

        for rank, macro in enumerate(macro_order):
            macro_mask = macro_labels == macro
            macro_size = int(macro_mask.sum())

            # La macro-regione a più bassa intensità (rank 0) = necrosi → NON dividere
            # Le macro-regioni attive (rank >= 1) → dividere in sottoregioni
            if self.split_mode == "active_only":
                should_split = (rank >= 1) and (macro_size > 100)
            else:
                should_split = macro_size > 100

            if not should_split:
                final_labels[macro_mask] = next_label
                sub_info[next_label] = {"macro": int(macro), "sub": 0,
                                        "size": macro_size, "rank": rank}
                next_label += 1
            else:
                X_macro = X[macro_mask]
                k_sub = min(self.n_sub, max(1, macro_size // 50))
                if k_sub == 1:
                    final_labels[macro_mask] = next_label
                    next_label += 1
                    continue
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    gmm2 = GaussianMixture(
                        n_components=k_sub, covariance_type=self.covariance_type,
                        n_init=self.n_init, random_state=42,
                    )
                    sub_labels = gmm2.fit_predict(X_macro)
                sub_means = {s: X_macro[sub_labels == s, primary_idx].mean()
                             for s in np.unique(sub_labels)}
                sub_order = sorted(sub_means, key=sub_means.get)
                remap = {old: i for i, old in enumerate(sub_order)}
                macro_indices = np.where(macro_mask)[0]
                for old_s in sub_order:
                    sel = np.zeros(n, dtype=bool)
                    sel[macro_indices[sub_labels == old_s]] = True
                    final_labels[sel] = next_label
                    sub_info[next_label] = {"macro": int(macro),
                                            "sub": remap[old_s],
                                            "size": int(sel.sum()), "rank": rank}
                    next_label += 1

        n_clusters = next_label
        log.info(f"Hierarchical [{context.modality.value}]: "
                 f"{self.n_level1} macro-regions → {n_clusters} final clusters")

        extra = {
            "best_k": n_clusters,
            "n_level1": self.n_level1,
            "primary_weight": self.primary_weight,
            "hierarchy": {str(k): v for k, v in sub_info.items()},
        }
        return final_labels, None, extra

    def _get_params(self) -> dict:
        return {
            "primary_weight": self.primary_weight,
            "n_level1": self.n_level1,
            "n_sub": self.n_sub,
        }
