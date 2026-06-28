"""
io_data/loader.py — Loader multi-modale per GlioCore.

Riconosce automaticamente la modalità in base ai file presenti:
  PET      — SUVR + SUV + mask  (T1 opzionale)
  MRI      — T1 + T1ce + T2 + FLAIR + seg  (BraTS)
  PET_MRI  — SUVR + SUV + T1 + mask  (+ T1ce/T2/FLAIR opzionali)

Nessun hack: niente "finto SUVR" da T1. Ogni modalità carica i suoi dati reali.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path

import nibabel as nib
import numpy as np

from io_data.modality import Modality, build_feature_set, FeatureSet, SegmentationContext

log = logging.getLogger(__name__)


class LoadError(RuntimeError):
    """Errore di caricamento con messaggio leggibile."""


# ── Pattern nomi file per canale (glob) ──────────────────────────────────────
CHANNEL_PATTERNS = {
    "suvr": ["SUVR_2_T1_cerebWM.nii.gz", "SUVR_2_T1_cerebWM.nii",
             "*suvr*.nii.gz", "*suvr*.nii", "*SUVR*.nii.gz", "*SUVR*.nii"],
    "suv":  ["SUV_2_T1.nii.gz", "SUV_2_T1.nii",
             "*suv*.nii.gz", "*suv*.nii", "*SUV*.nii.gz", "*SUV*.nii"],
    "mask": ["tumour_mask_4t.nii.gz", "tumour_mask_4t.nii",
             "*tumour_mask*.nii.gz", "*tumour_mask*.nii",
             "*tumor_mask*.nii.gz", "*tumor_mask*.nii",
             "*mask*.nii.gz", "*mask*.nii"],
    "t1":   ["T1.nii.gz", "T1.nii", "*-t1n.nii.gz", "*-t1n.nii",
             "*_t1.nii.gz", "*_t1.nii", "t1_w_brain.nii.gz", "t1_w_brain.nii",
             "T1w.nii.gz", "T1w.nii", "*t1n.nii.gz", "*t1n.nii"],
    "t1c":  ["*-t1c.nii.gz", "*-t1c.nii", "*_t1ce.nii.gz", "*_t1ce.nii",
             "T1ce.nii.gz", "T1ce.nii", "*t1c.nii.gz", "*t1c.nii"],
    "t2w":  ["*-t2w.nii.gz", "*-t2w.nii", "*_t2.nii.gz", "*_t2.nii",
             "T2.nii.gz", "T2.nii", "*t2w.nii.gz", "*t2w.nii"],
    "t2f":  ["*-t2f.nii.gz", "*-t2f.nii", "*_flair.nii.gz", "*_flair.nii",
             "FLAIR.nii.gz", "FLAIR.nii", "*t2f.nii.gz", "*t2f.nii"],
    "seg":  ["*-seg.nii.gz", "*-seg.nii", "*_seg.nii.gz", "*_seg.nii",
             "seg.nii.gz", "seg.nii", "segmentation.nii.gz", "segmentation.nii"],
}


@dataclass
class PatientData:
    """
    Struttura dati pulita e modality-aware.

    Sostituisce la vecchia PatientVolumes.
    Contiene i volumi grezzi per canale + metadati.
    """
    patient_id: str
    modality:   Modality
    volumes:    dict[str, np.ndarray]          # canale → volume 3D float32
    images:     dict[str, nib.Nifti1Image]     # canale → oggetto NIfTI
    mask:       np.ndarray                      # volume 3D bool

    @property
    def affine(self) -> np.ndarray:
        # Usa l'affine del primo canale disponibile
        first = next(iter(self.images.values()))
        return first.affine

    @property
    def shape(self) -> tuple:
        return self.mask.shape

    @property
    def n_tumour_voxels(self) -> int:
        return int(self.mask.sum())

    @property
    def reference_img(self) -> nib.Nifti1Image:
        """Immagine di riferimento per salvare output con affine corretto."""
        for ch in ["suvr", "t1", "t1c"]:
            if ch in self.images:
                return self.images[ch]
        return next(iter(self.images.values()))

    @property
    def available_channels(self) -> list[str]:
        return list(self.volumes.keys())

    def get_background(self) -> tuple[np.ndarray, str]:
        """Volume di sfondo per la visualizzazione e suo nome."""
        for ch in ["t1", "t1c", "suvr"]:
            if ch in self.volumes:
                return self.volumes[ch], ch.upper()
        first = next(iter(self.volumes.items()))
        return first[1], first[0].upper()

    def build_features(
        self,
        primary_channel: str | None = None,
        normalize: bool = True,
    ) -> FeatureSet:
        """Costruisce la FeatureSet per i modelli di segmentazione."""
        return build_feature_set(
            self.volumes, self.mask, self.modality,
            primary_channel=primary_channel, normalize=normalize,
        )

    def build_context(
        self,
        primary_channel: str | None = None,
        cluster_order: str = "ascending",
    ) -> SegmentationContext:
        """Costruisce il contesto per i modelli (volumi completi, affine)."""
        from io_data.modality import DEFAULT_PRIMARY_FEATURE
        pc = primary_channel or DEFAULT_PRIMARY_FEATURE[self.modality]
        if pc not in self.volumes:
            pc = self.available_channels[0]
        return SegmentationContext(
            modality=self.modality,
            full_volumes=self.volumes,
            affine=self.affine,
            primary_channel=pc,
            cluster_order=cluster_order,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTION E LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def detect_modality(patient_dir: Path) -> Modality:
    """
    Determina la modalità in base ai file presenti.

    Logica:
      ha SUVR+SUV+mask + (T1ce o T2 o FLAIR)  → PET_MRI
      ha SUVR+SUV+mask                        → PET
      ha T1+T1ce+T2+FLAIR+seg                 → MRI
    """
    found = {ch: _find_channel(patient_dir, ch) is not None
             for ch in CHANNEL_PATTERNS}

    has_pet  = found["suvr"] and found["suv"] and found["mask"]
    has_mri_full = found["t1"] and found["t1c"] and found["t2w"] and found["t2f"]
    has_mri_extra = found["t1c"] or found["t2w"] or found["t2f"]

    if has_pet and has_mri_extra and found["t1"]:
        return Modality.PET_MRI
    elif has_pet:
        return Modality.PET
    elif has_mri_full and found["seg"]:
        return Modality.MRI
    elif found["t1"] and found["seg"]:
        # MRI parziale (almeno T1 + seg)
        return Modality.MRI
    else:
        raise LoadError(
            f"Unable to determine the modality in {patient_dir.name}.\n"
            f"Channels found: {[ch for ch, ok in found.items() if ok]}\n"
            f"PET requires SUVR+SUV+mask. MRI requires T1+seg."
        )


def load_patient(
    patient_dir: Path,
    file_cfg: dict | None = None,
    force_modality: Modality | None = None,
) -> PatientData:
    """
    Carica un paziente con riconoscimento automatico della modalità.

    Parameters
    ----------
    patient_dir    : cartella del paziente
    file_cfg       : opzionale, override nomi file espliciti per canale
    force_modality : forza una modalità invece di rilevarla
    """
    patient_dir = Path(patient_dir)
    if not patient_dir.is_dir():
        raise LoadError(f"Folder not found: {patient_dir}")

    patient_id = patient_dir.name
    modality   = force_modality or detect_modality(patient_dir)

    log.info(f"[{patient_id}] Detected modality: {modality.value}")

    if modality == Modality.PET:
        return _load_pet(patient_dir, patient_id, file_cfg)
    elif modality == Modality.MRI:
        return _load_mri(patient_dir, patient_id)
    elif modality == Modality.PET_MRI:
        return _load_pet_mri(patient_dir, patient_id, file_cfg)
    else:
        raise LoadError(f"Unsupported modality: {modality}")


# ── Loader per modalità ───────────────────────────────────────────────────────

def _load_pet(patient_dir, patient_id, file_cfg) -> PatientData:
    """PET: SUVR + SUV + mask obbligatori, T1 opzionale."""
    cfg = file_cfg or {}

    suvr_img = _load_channel(patient_dir, "suvr", cfg, required=True, label="SUVR")
    suv_img  = _load_channel(patient_dir, "suv",  cfg, required=True, label="SUV")
    mask_img = _load_channel(patient_dir, "mask", cfg, required=True, label="mask")

    volumes = {
        "suvr": _to_float32(suvr_img),
        "suv":  _to_float32(suv_img),
    }
    images = {"suvr": suvr_img, "suv": suv_img, "mask": mask_img}

    mask = np.asarray(mask_img.dataobj) > 0

    # T1 opzionale
    t1_img = _load_channel(patient_dir, "t1", cfg, required=False, label="T1")
    if t1_img is not None:
        t1 = _to_float32(t1_img)
        if t1.shape == volumes["suvr"].shape:
            volumes["t1"] = t1
            images["t1"]  = t1_img
        else:
            log.warning(f"[{patient_id}] T1 shape mismatch, ignored")

    _validate(patient_id, volumes, mask)
    return PatientData(patient_id, Modality.PET, volumes, images, mask)


def _load_mri(patient_dir, patient_id) -> PatientData:
    """MRI BraTS: T1 + T1ce + T2 + FLAIR + seg."""
    t1_img  = _load_channel(patient_dir, "t1",  {}, required=True,  label="T1")
    seg_img = _load_channel(patient_dir, "seg", {}, required=True,  label="seg")
    t1c_img = _load_channel(patient_dir, "t1c", {}, required=False, label="T1ce")
    t2w_img = _load_channel(patient_dir, "t2w", {}, required=False, label="T2")
    t2f_img = _load_channel(patient_dir, "t2f", {}, required=False, label="FLAIR")

    volumes = {"t1": _to_float32(t1_img)}
    images  = {"t1": t1_img, "seg": seg_img}

    for ch, img in [("t1c", t1c_img), ("t2w", t2w_img), ("t2f", t2f_img)]:
        if img is not None:
            arr = _to_float32(img)
            if arr.shape == volumes["t1"].shape:
                volumes[ch] = arr
                images[ch]  = img
            else:
                log.warning(f"[{patient_id}] {ch} shape mismatch, ignored")

    # In MRI la maschera è seg > 0 (tutto il tumore)
    seg_data = np.asarray(seg_img.dataobj)
    mask = seg_data > 0

    # Salva anche la seg grezza per il benchmark (sottoregioni)
    volumes["_seg_raw"] = seg_data.astype(np.float32)

    _validate(patient_id, {k: v for k, v in volumes.items() if not k.startswith("_")}, mask)
    return PatientData(patient_id, Modality.MRI, volumes, images, mask)


def _load_pet_mri(patient_dir, patient_id, file_cfg) -> PatientData:
    """PET+MRI: SUVR+SUV+T1+mask obbligatori, T1ce/T2/FLAIR opzionali."""
    cfg = file_cfg or {}

    suvr_img = _load_channel(patient_dir, "suvr", cfg, required=True, label="SUVR")
    suv_img  = _load_channel(patient_dir, "suv",  cfg, required=True, label="SUV")
    t1_img   = _load_channel(patient_dir, "t1",   cfg, required=True, label="T1")
    mask_img = _load_channel(patient_dir, "mask", cfg, required=True, label="mask")

    volumes = {
        "suvr": _to_float32(suvr_img),
        "suv":  _to_float32(suv_img),
        "t1":   _to_float32(t1_img),
    }
    images = {"suvr": suvr_img, "suv": suv_img, "t1": t1_img, "mask": mask_img}
    ref_shape = volumes["suvr"].shape

    # Canali MRI opzionali
    for ch, label in [("t1c", "T1ce"), ("t2w", "T2"), ("t2f", "FLAIR")]:
        img = _load_channel(patient_dir, ch, cfg, required=False, label=label)
        if img is not None:
            arr = _to_float32(img)
            if arr.shape == ref_shape:
                volumes[ch] = arr
                images[ch]  = img
            else:
                log.warning(f"[{patient_id}] {label} shape mismatch, ignored")

    mask = np.asarray(mask_img.dataobj) > 0
    _validate(patient_id, volumes, mask)
    return PatientData(patient_id, Modality.PET_MRI, volumes, images, mask)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_channel(patient_dir: Path, channel: str,
                  cfg: dict | None = None) -> Path | None:
    """Trova il file di un canale provando override + pattern."""
    cfg = cfg or {}
    # Override esplicito
    if channel in cfg and cfg[channel]:
        p = patient_dir / cfg[channel]
        if p.exists():
            return p
    # Pattern glob
    for pattern in CHANNEL_PATTERNS.get(channel, []):
        if "*" in pattern:
            matches = sorted(patient_dir.glob(pattern))
            if matches:
                return matches[0]
        else:
            p = patient_dir / pattern
            if p.exists():
                return p
    return None


def _load_channel(patient_dir, channel, cfg, required, label) -> nib.Nifti1Image | None:
    path = _find_channel(patient_dir, channel, cfg)
    if path is None:
        if required:
            available = [f.name for f in patient_dir.glob("*.nii.gz")]
            raise LoadError(
                f"File {label} (channel '{channel}') not found in {patient_dir.name}.\n"
                f"Available files: {available}"
            )
        return None
    return _load_nifti(path, label)


def _load_nifti(path: Path, label: str) -> nib.Nifti1Image:
    # Caricamento lazy: i dati vengono letti una sola volta da _to_float32
    # (np.asarray(dataobj, float32)). Evitiamo il get_fdata() che caricava
    # l'intero volume in float64 solo per scartarlo — stessi valori, metà I/O
    # e memoria. Si valida l'header (shape) senza materializzare i dati.
    try:
        img = nib.load(str(path))
        _ = img.shape   # forza il parsing dell'header, niente lettura dati
        return img
    except Exception as e:
        raise LoadError(f"Unable to read {label} ({path.name}): {e}") from e


def _to_float32(img: nib.Nifti1Image) -> np.ndarray:
    return np.asarray(img.dataobj, dtype=np.float32)


def _validate(patient_id: str, volumes: dict, mask: np.ndarray) -> None:
    """Valida shape coerenti e maschera non vuota."""
    shapes = {ch: v.shape for ch, v in volumes.items()}
    ref_shape = next(iter(shapes.values()))
    for ch, shp in shapes.items():
        if shp != ref_shape:
            raise LoadError(
                f"[{patient_id}] Shape mismatch: {ch} {shp} vs reference {ref_shape}"
            )
    if mask.shape != ref_shape:
        raise LoadError(
            f"[{patient_id}] Mask shape {mask.shape} != volumes {ref_shape}"
        )
    n = int(mask.sum())
    if n == 0:
        raise LoadError(f"[{patient_id}] Empty tumor mask")
    log.info(f"[{patient_id}] Validated — {n:,} voxels, channels: {list(volumes.keys())}")


# ── Funzioni di compatibilità ─────────────────────────────────────────────────

def load_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Carica un singolo NIfTI → (img, array float32)."""
    img = _load_nifti(Path(path), Path(path).name)
    return img, _to_float32(img)


def save_nifti_like(reference, data, path, dtype=np.uint8) -> None:
    """Salva un array come NIfTI con affine/header del riferimento."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(dtype), reference.affine, reference.header)
    nib.save(img, str(path))
    log.debug(f"Saved: {path}")
