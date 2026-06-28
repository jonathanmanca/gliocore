"""
GlioCore – configurazione globale.
Modifica questo file per adattare il tool al tuo ambiente.
"""
import os
from pathlib import Path

# ── Percorsi principali ──────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent.parent
DATA_DIR     = BASE_DIR / "data" / "patients"   # una sottocartella per paziente
OUTPUT_DIR   = BASE_DIR / "data" / "output"
ATLAS_DIR    = BASE_DIR / "data" / "atlas"      # JHU labels + MNI152 template
DB_PATH      = BASE_DIR / "data" / "sessions.db"

# ── Nomi file attesi dentro ogni cartella paziente ───────────────────────────
FILE_SUVR  = "SUVR_2_T1_cerebWM.nii"
FILE_SUV   = "SUV_2_T1.nii"
FILE_MASK  = "tumour_mask_4t.nii"
FILE_T1    = "T1.nii.gz"          # anatomico per visualizzazione background

# ── Parametri GMM ────────────────────────────────────────────────────────────
GMM_K_MIN        = 2
GMM_K_MAX        = 6
GMM_CRITERION    = "BIC"          # "BIC" | "AIC"
GMM_N_INIT       = 10
GMM_MIN_VOXELS   = 50

# ── Parametri Fuzzy C-Means ──────────────────────────────────────────────────
FCM_K_DEFAULT    = 3
FCM_M            = 2.0            # fuzziness exponent
FCM_ERROR        = 1e-5
FCM_MAXITER      = 300

# ── Parametri Level Set ──────────────────────────────────────────────────────
LS_NUM_ITER      = 200
LS_SMOOTHING     = 3

# ── Atlas ────────────────────────────────────────────────────────────────────
MNI152_TEMPLATE  = ATLAS_DIR / "MNI152_T1_1mm.nii.gz"
JHU_LABELS       = ATLAS_DIR / "JHU-ICBM-labels-1mm.nii.gz"
JHU_LOOKUP       = ATLAS_DIR / "JHU_labels.csv"   # colonne: index, name, abbreviation

# ── Agenti AI ────────────────────────────────────────────────────────────────
# La chiave NON va scritta qui: si legge dall'ambiente per non finire nel repo.
#   Windows (PowerShell):  $env:ANTHROPIC_API_KEY = "sk-ant-..."
#   Linux/macOS:           export ANTHROPIC_API_KEY="sk-ant-..."
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MODEL       = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
AGENT_MAX_TOKENS      = 1500

# ── UI ───────────────────────────────────────────────────────────────────────
APP_TITLE        = "GlioCore — Glioma Segmentation Tool"
COLORMAP_BASE    = "gray"
COLORMAP_MASK    = "red"
COLORMAP_CLUSTER = "turbo"
OVERLAY_OPACITY  = 0.45

# ── Colori cluster (label 1 = hypo/necrosi, label k = hyper/infiltrazione) ──
CLUSTER_COLORS = {
    "hypo":    "#2196F3",   # blu  – ipometabolismo / necrosi
    "mid":     "#4CAF50",   # verde – metabolismo intermedio
    "hyper":   "#F44336",   # rosso – ipermetabolismo / infiltrazione
}
