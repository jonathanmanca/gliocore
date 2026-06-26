"""
io_data/modality.py — Definizione modalità e costruzione feature set.

Tre modalità esplicite:
  PET      — SUVR, SUV, tumor mask (T1 opzionale)
  MRI      — T1, T1ce, T2, FLAIR, seg (BraTS)
  PET_MRI  — SUVR, SUV, T1, tumor mask obbligatori; T1ce, T2, FLAIR opzionali

Questo modulo NON usa hack: ogni modalità ha le sue feature reali.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


class Modality(str, Enum):
    PET     = "PET"
    MRI     = "MRI"
    PET_MRI = "PET_MRI"


# Canali possibili per ciascuna modalità, in ordine di priorità
PET_CHANNELS     = ["suvr", "suv"]
MRI_CHANNELS     = ["t1", "t1c", "t2w", "t2f"]
PET_MRI_CHANNELS = ["suvr", "suv", "t1", "t1c", "t2w", "t2f"]

# Feature primaria di default per il cluster ordering (per modalità)
DEFAULT_PRIMARY_FEATURE = {
    Modality.PET:     "suvr",
    Modality.MRI:     "t1c",     # T1ce è il canale più informativo per gliomi
    Modality.PET_MRI: "suvr",
}


@dataclass
class FeatureSet:
    """
    Matrice di feature pronta per i modelli di segmentazione.

    matrix       : (N_voxel, n_features) — solo voxel dentro la maschera
    channel_names: nome di ogni colonna (es. ['suvr', 'suv'])
    mask         : volume 3D booleano
    shape        : shape del volume completo
    modality     : modalità di provenienza
    primary_idx  : indice colonna della feature primaria (per ordering)
    normalized   : se la matrice è già normalizzata
    """
    matrix:        np.ndarray
    channel_names: list[str]
    mask:          np.ndarray
    shape:         tuple
    modality:      Modality
    primary_idx:   int = 0
    normalized:    bool = False

    @property
    def n_features(self) -> int:
        return self.matrix.shape[1]

    @property
    def n_voxels(self) -> int:
        return self.matrix.shape[0]

    @property
    def primary_feature(self) -> np.ndarray:
        """Vettore 1D della feature primaria (per cluster ordering)."""
        return self.matrix[:, self.primary_idx]

    def set_primary(self, channel_name: str) -> None:
        """Imposta quale canale usare come feature primaria per l'ordering."""
        if channel_name not in self.channel_names:
            raise ValueError(
                f"Channel '{channel_name}' not available. "
                f"Available: {self.channel_names}"
            )
        self.primary_idx = self.channel_names.index(channel_name)

    def normalize(self) -> "FeatureSet":
        """Restituisce una copia normalizzata (z-score per canale)."""
        if self.normalized:
            return self
        scaler = StandardScaler()
        norm_matrix = scaler.fit_transform(self.matrix)
        return FeatureSet(
            matrix=norm_matrix.astype(np.float32),
            channel_names=self.channel_names,
            mask=self.mask,
            shape=self.shape,
            modality=self.modality,
            primary_idx=self.primary_idx,
            normalized=True,
        )

    def single_channel(self, channel_name: str | None = None) -> np.ndarray:
        """
        Restituisce un singolo canale come (N, 1) — per modelli 1D
        come Level Set che lavorano su una sola intensità.
        """
        idx = (self.channel_names.index(channel_name)
               if channel_name else self.primary_idx)
        return self.matrix[:, idx].reshape(-1, 1)


@dataclass
class SegmentationContext:
    """
    Contesto passato ai modelli oltre alle feature.
    Contiene i volumi completi e info di modalità per modelli che
    ne hanno bisogno (Level Set richiede il volume 3D, MRF la connettività).
    """
    modality:      Modality
    full_volumes:  dict[str, np.ndarray] = field(default_factory=dict)
    affine:        np.ndarray | None = None
    primary_channel: str = "suvr"
    # ordering: 'ascending' = cluster 1 ha valore primario più basso
    cluster_order: str = "ascending"

    def get_volume(self, channel: str) -> np.ndarray | None:
        return self.full_volumes.get(channel)

    @property
    def primary_volume(self) -> np.ndarray | None:
        return self.full_volumes.get(self.primary_channel)


def build_feature_set(
    volumes:    dict[str, np.ndarray],   # canale → volume 3D
    mask:       np.ndarray,              # volume 3D bool
    modality:   Modality,
    primary_channel: str | None = None,
    normalize:  bool = True,
) -> FeatureSet:
    """
    Costruisce una FeatureSet dai volumi disponibili.

    Usa solo i canali presenti in `volumes` tra quelli previsti per la modalità.
    Ignora in modo pulito i canali mancanti (nessun crash, nessun finto dato).
    """
    # Canali previsti per la modalità
    expected = {
        Modality.PET:     PET_CHANNELS,
        Modality.MRI:     MRI_CHANNELS,
        Modality.PET_MRI: PET_MRI_CHANNELS,
    }[modality]

    # Usa solo i canali realmente disponibili
    available = [ch for ch in expected if ch in volumes and volumes[ch] is not None]

    if not available:
        raise ValueError(
            f"No channel available for modality {modality.value}. "
            f"Expected: {expected}, provided: {list(volumes.keys())}"
        )

    # Costruisci la matrice impilando i canali sui voxel mascherati
    columns = [volumes[ch][mask] for ch in available]
    matrix  = np.column_stack(columns).astype(np.float32)

    # Determina la feature primaria
    if primary_channel is None:
        primary_channel = DEFAULT_PRIMARY_FEATURE[modality]
    # Se la primaria scelta non è disponibile, usa la prima disponibile
    if primary_channel not in available:
        log.warning(
            f"Primary feature '{primary_channel}' not available for "
            f"{modality.value}, using '{available[0]}'"
        )
        primary_channel = available[0]
    primary_idx = available.index(primary_channel)

    fs = FeatureSet(
        matrix=matrix,
        channel_names=available,
        mask=mask,
        shape=mask.shape,
        modality=modality,
        primary_idx=primary_idx,
        normalized=False,
    )

    log.info(
        f"FeatureSet [{modality.value}]: {fs.n_voxels:,} voxels × "
        f"{fs.n_features} features {available}, primary='{primary_channel}'"
    )

    return fs.normalize() if normalize else fs


def order_clusters_by_feature(
    labels:        np.ndarray,    # (N,) etichette 0-indexed o 1-indexed
    feature_1d:    np.ndarray,    # (N,) valori della feature primaria
    order:         str = "ascending",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Riordina le etichette dei cluster in base alla media della feature primaria.

    Returns:
      new_labels : etichette rimappate 1..k (1 = valore primario più basso se ascending)
      sort_idx   : permutazione applicata ai cluster originali
    """
    unique = np.unique(labels)
    unique = unique[unique >= 0]   # ignora -1 (outlier DBSCAN)

    means = {c: float(feature_1d[labels == c].mean()) for c in unique}
    sorted_clusters = sorted(unique, key=lambda c: means[c],
                             reverse=(order == "descending"))

    remap = {old: new + 1 for new, old in enumerate(sorted_clusters)}
    # outlier -1 → 0 (fuori cluster)
    new_labels = np.array([remap.get(l, 0) for l in labels], dtype=np.uint8)
    sort_idx = np.array(sorted_clusters)

    return new_labels, sort_idx
