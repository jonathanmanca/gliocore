"""
atlas/wm_report.py — Analisi di overlap con atlante JHU di materia bianca.

Dato una maschera tumorale in spazio MNI152, calcola la percentuale di
overlap con ogni fascio dell'atlante JHU e genera un report strutturato.

Riferimento: Hua K. et al. (2008) NeuroImage
             DOI: 10.1016/j.neuroimage.2007.07.053
"""
from __future__ import annotations
import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import nibabel as nib

from config.settings import JHU_LABELS, JHU_LOOKUP, ATLAS_DIR

log = logging.getLogger(__name__)

# Nomi JHU built-in se il CSV di lookup non è disponibile
_JHU_NAMES_BUILTIN = {
    1:  "Corpus callosum (genu)",
    2:  "Corpus callosum (body)",
    3:  "Corpus callosum (splenium)",
    4:  "Forceps minor",
    5:  "Forceps major",
    6:  "Superior longitudinal fasciculus (L)",
    7:  "Superior longitudinal fasciculus (R)",
    8:  "Anterior thalamic radiation (L)",
    9:  "Anterior thalamic radiation (R)",
    10: "Uncinate fasciculus (L)",
    11: "Uncinate fasciculus (R)",
    12: "Cingulum (cingulate gyrus) (L)",
    13: "Cingulum (cingulate gyrus) (R)",
    14: "Cingulum (parahippocampal) (L)",
    15: "Cingulum (parahippocampal) (R)",
    16: "Corticospinal tract (L)",
    17: "Corticospinal tract (R)",
    18: "Corticorubral tract (L)",
    19: "Corticorubral tract (R)",
    20: "Middle cerebellar peduncle",
    21: "Medial pontine tract",
    22: "Medial longitudinal fasciculus",
    23: "Superior cerebellar peduncle (L)",
    24: "Superior cerebellar peduncle (R)",
    25: "Cerebral peduncle (L)",
    26: "Cerebral peduncle (R)",
    27: "Corticostriatal tract (L)",
    28: "Corticostriatal tract (R)",
    29: "Internal capsule — anterior limb (L)",
    30: "Internal capsule — anterior limb (R)",
    31: "Internal capsule — genu (L)",
    32: "Internal capsule — genu (R)",
    33: "Internal capsule — posterior limb (L)",
    34: "Internal capsule — posterior limb (R)",
    35: "Internal capsule — sublenticular (L)",
    36: "Internal capsule — sublenticular (R)",
    37: "Internal capsule — retrolenticular (L)",
    38: "Internal capsule — retrolenticular (R)",
    39: "Anterior corona radiata (L)",
    40: "Anterior corona radiata (R)",
    41: "Superior corona radiata (L)",
    42: "Superior corona radiata (R)",
    43: "Posterior corona radiata (L)",
    44: "Posterior corona radiata (R)",
    45: "Optic radiation (L)",
    46: "Optic radiation (R)",
    47: "Inferior longitudinal fasciculus (L)",
    48: "Inferior longitudinal fasciculus (R)",
}

# Fasci funzionalmente critici — segnalati in rosso nel report
_CRITICAL_TRACTS = {
    16, 17,   # Tratto cortico-spinale
    6, 7,     # Fascicolo longitudinale superiore
    10, 11,   # Fascicolo uncinato
    45, 46,   # Radiazione ottica
    12, 13,   # Cingolo
}


@dataclass
class TractOverlap:
    """Overlap tra la maschera tumorale e un fascio WM."""
    tract_id:     int
    tract_name:   str
    n_voxels:     int           # voxel della maschera che cadono nel fascio
    tract_size:   int           # dimensione totale del fascio nell'atlante
    overlap_pct:  float         # n_voxels / tract_size * 100
    is_critical:  bool          # fascio funzionalmente critico


@dataclass
class WMReport:
    """Report completo di overlap per un paziente."""
    patient_id:   str
    model_name:   str
    n_clusters:   int
    tracts:       list[TractOverlap] = field(default_factory=list)
    total_mask_voxels: int = 0

    @property
    def affected_tracts(self) -> list[TractOverlap]:
        """Solo i fasci con overlap > 0."""
        return [t for t in self.tracts if t.n_voxels > 0]

    @property
    def critical_affected(self) -> list[TractOverlap]:
        """Fasci critici coinvolti."""
        return [t for t in self.affected_tracts if t.is_critical]

    def to_dict_list(self) -> list[dict]:
        return [
            {
                "Tract":            t.tract_name,
                "Voxels involved":  t.n_voxels,
                "Tract size":       t.tract_size,
                "Overlap %":        round(t.overlap_pct, 2),
                "Critical":         "⚠️ YES" if t.is_critical else "no",
            }
            for t in self.affected_tracts
        ]

    def summary_text(self) -> str:
        """Testo di sintesi per l'agente AI e il referto."""
        if not self.affected_tracts:
            return "No white-matter tract involved by the lesion."

        lines = [
            f"WM tracts involved: {len(self.affected_tracts)} "
            f"(out of {len(self.tracts)} in the JHU atlas)",
        ]
        if self.critical_affected:
            names = ", ".join(t.tract_name for t in self.critical_affected)
            lines.append(f"⚠️  Critical tracts affected: {names}")

        top3 = sorted(self.affected_tracts, key=lambda t: t.overlap_pct, reverse=True)[:3]
        lines.append("Greatest involvement:")
        for t in top3:
            lines.append(f"  • {t.tract_name}: {t.overlap_pct:.1f}% of the tract")

        return "\n".join(lines)


class WMAtlasAnalyzer:
    """
    Analizza l'overlap tra maschera tumorale (in spazio MNI) e atlante JHU.

    Utilizzo:
        analyzer = WMAtlasAnalyzer()
        report   = analyzer.analyze(mask_mni_path, patient_id, model_name, k)
        df       = pd.DataFrame(report.to_dict_list())
    """

    def __init__(self, atlas_path: Path = JHU_LABELS, lookup_path: Path = JHU_LOOKUP):
        self.atlas_path  = Path(atlas_path)
        self.lookup_path = Path(lookup_path)
        self._atlas_data = None
        self._names      = None
        self._load_atlas()

    def analyze(
        self,
        mask_mni_path: Path,
        patient_id:    str,
        model_name:    str,
        n_clusters:    int,
    ) -> WMReport:
        """
        Calcola l'overlap tra maschera (in spazio MNI) e ogni fascio JHU.

        Parameters
        ----------
        mask_mni_path : maschera tumorale già registrata in spazio MNI152
        """
        if self._atlas_data is None:
            raise RuntimeError(
                "JHU atlas not available. "
                f"Download JHU-ICBM-labels-1mm.nii.gz and place it in {ATLAS_DIR}"
            )

        mask_mni_path = Path(mask_mni_path)
        if not mask_mni_path.exists():
            raise FileNotFoundError(f"MNI mask not found: {mask_mni_path}")

        mask_img  = nib.load(str(mask_mni_path))
        mask_data = np.asarray(mask_img.dataobj) > 0

        # Verifica che le shape corrispondano
        if mask_data.shape != self._atlas_data.shape:
            log.warning(
                f"Shape mismatch: mask {mask_data.shape} vs atlas {self._atlas_data.shape}. "
                "The registration may not be correct."
            )

        report = WMReport(
            patient_id=patient_id,
            model_name=model_name,
            n_clusters=n_clusters,
            total_mask_voxels=int(mask_data.sum()),
        )

        # Per ogni fascio calcola l'overlap
        unique_labels = [l for l in np.unique(self._atlas_data) if l > 0]

        for tract_id in unique_labels:
            tract_mask  = self._atlas_data == tract_id
            tract_size  = int(tract_mask.sum())
            overlap     = int((mask_data & tract_mask).sum())

            if tract_size == 0:
                continue

            overlap_pct = overlap / tract_size * 100
            tract_name  = self._names.get(int(tract_id), f"Tract #{tract_id}")

            report.tracts.append(TractOverlap(
                tract_id=int(tract_id),
                tract_name=tract_name,
                n_voxels=overlap,
                tract_size=tract_size,
                overlap_pct=round(overlap_pct, 3),
                is_critical=int(tract_id) in _CRITICAL_TRACTS,
            ))

        # Ordina per overlap decrescente
        report.tracts.sort(key=lambda t: t.overlap_pct, reverse=True)

        n_affected = len(report.affected_tracts)
        log.info(
            f"[{patient_id}] WM atlas: {n_affected} tracts involved "
            f"({len(report.critical_affected)} critical)"
        )
        return report

    def save_report_csv(self, report: WMReport, output_path: Path) -> None:
        """Salva il report in formato CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rows = report.to_dict_list()
        if not rows:
            log.info("No tract involved — CSV not generated")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"WM report saved: {output_path}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_atlas(self) -> None:
        """Carica l'atlante JHU e il lookup dei nomi."""
        if not self.atlas_path.exists():
            log.warning(
                f"JHU atlas not found: {self.atlas_path}\n"
                "Download it from: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/Atlases\n"
                f"Expected file: {self.atlas_path}"
            )
            return

        try:
            atlas_img        = nib.load(str(self.atlas_path))
            self._atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int16)
            log.info(f"JHU atlas loaded: {self.atlas_path} — shape {self._atlas_data.shape}")
        except Exception as e:
            log.error(f"Atlas loading error: {e}")
            return

        # Carica nomi da CSV o usa builtin
        self._names = _JHU_NAMES_BUILTIN.copy()
        if self.lookup_path.exists():
            try:
                with open(self.lookup_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        idx  = int(row.get("index", row.get("id", 0)))
                        name = row.get("name", row.get("label", f"Tract #{idx}"))
                        self._names[idx] = name
                log.info(f"JHU lookup loaded: {len(self._names)} tracts")
            except Exception as e:
                log.warning(f"Unable to read lookup CSV: {e} — using builtin names")
