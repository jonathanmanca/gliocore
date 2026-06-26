"""
atlas/registration.py — Registrazione T1 paziente → spazio MNI152 con ANTsPy.

Pipeline:
1. Registrazione affine + SyN (deformabile): T1-paziente → MNI152
2. Applicazione della stessa trasformazione alla maschera tumorale
3. La maschera in spazio MNI può essere confrontata con l'atlante JHU

Riferimento: Avants BB et al. (2011) NeuroImage — ANTs
             Thiebaut de Schotten et al. (2018) BCBtoolkit — GigaScience
"""
from __future__ import annotations
import logging
from pathlib import Path

import numpy as np
import nibabel as nib

from config.settings import MNI152_TEMPLATE, ATLAS_DIR

log = logging.getLogger(__name__)


class RegistrationError(RuntimeError):
    pass


class MNIRegistration:
    """
    Registra il T1 del paziente in spazio MNI152 e applica la trasformazione
    alla maschera tumorale.

    Richiede antspyx installato: pip install antspyx
    """

    def __init__(self, template_path: Path = MNI152_TEMPLATE):
        self.template_path = Path(template_path)
        self._check_template()

    def register(
        self,
        t1_path:    Path,
        mask_path:  Path,
        output_dir: Path,
        type_of_transform: str = "SyN",
    ) -> dict[str, Path]:
        """
        Esegue la registrazione e restituisce i percorsi dei file output.

        Parameters
        ----------
        t1_path    : T1 del paziente in spazio nativo
        mask_path  : maschera tumorale in spazio nativo (stesso spazio del T1)
        output_dir : dove salvare i file registrati
        type_of_transform : "Affine" (veloce) o "SyN" (accurato, ~5 min)

        Returns
        -------
        dict con chiavi:
          t1_mni       : T1 registrato in spazio MNI
          mask_mni     : maschera in spazio MNI
          transforms   : lista path delle trasformazioni (per uso futuro)
        """
        try:
            import ants
        except ImportError:
            raise RegistrationError(
                "ANTsPy not installed. Run: pip install antspyx\n"
                "Note: the download is ~1GB, it takes a few minutes."
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t1_path   = Path(t1_path)
        mask_path = Path(mask_path)

        log.info(f"Registration {t1_path.name} → MNI152 ({type_of_transform})")

        # Carica immagini come oggetti ANTs
        fixed  = ants.image_read(str(self.template_path))
        moving = ants.image_read(str(t1_path))

        # Registrazione
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform=type_of_transform,
            outprefix=str(output_dir / "reg_"),
            verbose=False,
        )

        # Salva T1 registrato
        t1_mni_path = output_dir / "T1_MNI152.nii.gz"
        ants.image_write(registration["warpedmovout"], str(t1_mni_path))
        log.info(f"T1 in MNI saved: {t1_mni_path}")

        # Applica la stessa trasformazione alla maschera (interpolazione nearest neighbor)
        mask_ants = ants.image_read(str(mask_path))
        mask_mni_ants = ants.apply_transforms(
            fixed=fixed,
            moving=mask_ants,
            transformlist=registration["fwdtransforms"],
            interpolator="nearestNeighbor",
        )

        mask_mni_path = output_dir / "mask_tumour_MNI152.nii.gz"
        ants.image_write(mask_mni_ants, str(mask_mni_path))
        log.info(f"Mask in MNI saved: {mask_mni_path}")

        return {
            "t1_mni":     t1_mni_path,
            "mask_mni":   mask_mni_path,
            "transforms": registration["fwdtransforms"],
        }

    def _check_template(self) -> None:
        if not self.template_path.exists():
            log.warning(
                f"MNI152 template not found: {self.template_path}\n"
                f"Download it from FSL or from: https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009\n"
                f"Save it in: {ATLAS_DIR}"
            )
