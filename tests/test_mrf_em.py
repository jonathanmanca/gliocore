"""
tests/test_mrf_em.py — garanzia di RIPRODUCIBILITÀ per il refactor di MRF-EM.

La vettorizzazione di `_icm_step` deve produrre etichette IDENTICHE al ciclo
scalare originale, altrimenti i numeri del benchmark BraTS cambierebbero.
Qui si confronta la nuova implementazione contro una copia fedele del codice
originale su input casuali, e si verifica il determinismo del fit completo.

Eseguibile con `pytest` o come script: `python tests/test_mrf_em.py`.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segmentation.mrf_em import MRFEMSegmentation, _NEIGHBORS_6
from io_data.modality import build_feature_set, Modality, SegmentationContext


def reference_icm_step(X, labels_vol, coords, mask, means, variances, weights, k, beta):
    """Copia fedele dell'_icm_step scalare ORIGINALE (la 'verità' da preservare)."""
    shape = labels_vol.shape
    new_labels = labels_vol[mask].copy()
    for i, (z, y, x) in enumerate(coords):
        xi = X[i]
        best_label, best_energy = new_labels[i], np.inf
        for c in range(k):
            diff = xi - means[c]
            data_e = 0.5 * np.sum(diff**2 / variances[c]) \
                     + 0.5 * np.sum(np.log(variances[c]))
            data_e -= np.log(max(weights[c], 1e-10))
            n_disc = 0
            for dz, dy, dx in _NEIGHBORS_6:
                nz, ny, nx = z+dz, y+dy, x+dx
                if (0 <= nz < shape[0] and 0 <= ny < shape[1]
                        and 0 <= nx < shape[2] and mask[nz, ny, nx]):
                    if labels_vol[nz, ny, nx] != c:
                        n_disc += 1
            energy = data_e + beta * n_disc
            if energy < best_energy:
                best_energy, best_label = energy, c
        new_labels[i] = best_label
    return new_labels


def _random_case(seed, shape, k, D):
    rng = np.random.RandomState(seed)
    mask = rng.rand(*shape) > 0.4              # ~60% acceso, con bordi e buchi
    n = int(mask.sum())
    coords = np.column_stack(np.where(mask))
    X = rng.randn(n, D).astype(np.float64)
    labels_vol = np.zeros(shape, dtype=np.int32)
    labels_vol[mask] = rng.randint(0, k, size=n)
    means = rng.randn(k, D)
    variances = rng.uniform(0.1, 1.5, size=(k, D))
    weights = rng.uniform(0.0, 1.0, size=k)    # include valori ~0 (testa max(.,1e-10))
    weights[0] = 1e-12
    return X, labels_vol, coords, mask, means, variances, weights


def test_icm_step_identical_to_scalar():
    for seed in range(8):
        for k in (2, 3, 4):
            for beta in (0.0, 1.5, 2.5):
                X, lv, coords, mask, means, var, w = _random_case(
                    seed, (8, 9, 7), k, D=4)
                model = MRFEMSegmentation(beta=beta, k=k)
                got = model._icm_step(X, lv, coords, mask, means, var, w, k)
                ref = reference_icm_step(X, lv, coords, mask, means, var, w, k, beta)
                assert np.array_equal(got, ref), (
                    f"divergenza ICM seed={seed} k={k} beta={beta}: "
                    f"{int((got != ref).sum())} voxel diversi")


def test_full_fit_deterministic():
    shape = (14, 14, 14)
    rng = np.random.RandomState(0)
    mask = np.zeros(shape, dtype=bool)
    mask[2:12, 2:12, 2:12] = True
    vols = {
        "t1":  rng.rand(*shape).astype("float32"),
        "t1c": rng.rand(*shape).astype("float32"),
        "t2w": rng.rand(*shape).astype("float32"),
        "t2f": rng.rand(*shape).astype("float32"),
    }
    fs = build_feature_set(vols, mask, Modality.MRI, normalize=True)
    ctx = SegmentationContext(modality=Modality.MRI, full_volumes=vols,
                              primary_channel="t1c")
    r1 = MRFEMSegmentation(beta=1.5, k=3).fit(fs, ctx)
    r2 = MRFEMSegmentation(beta=1.5, k=3).fit(fs, ctx)
    assert np.array_equal(r1.label_volume, r2.label_volume), "fit non deterministico"


def _run_all():
    test_icm_step_identical_to_scalar()
    print("✔ test_icm_step_identical_to_scalar (8 seed × 3 k × 3 beta)")
    test_full_fit_deterministic()
    print("✔ test_full_fit_deterministic")
    print("\nMRF-EM: vettorizzazione equivalente all'originale, risultati preservati.")


if __name__ == "__main__":
    _run_all()
