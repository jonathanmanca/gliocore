"""
learning/session_db.py — database SQLite per sessioni e correzioni.

Ogni volta che un utente corregge una maschera, la correzione viene salvata.
Il modulo di apprendimento attivo la usa per migliorare i modelli nel tempo.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    create_engine, Column, Integer, String,
    Float, DateTime, Text, Boolean,
)
from sqlalchemy.orm import DeclarativeBase, Session

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


class SegmentationRun(Base):
    """Un'esecuzione di segmentazione su un paziente."""
    __tablename__ = "segmentation_runs"

    id          = Column(Integer, primary_key=True)
    patient_id  = Column(String(64), nullable=False, index=True)
    model_name  = Column(String(32), nullable=False)
    n_clusters  = Column(Integer)
    params      = Column(Text)      # JSON
    metrics     = Column(Text)      # JSON
    output_dir  = Column(String(256))
    created_at  = Column(DateTime, default=datetime.utcnow)
    is_validated = Column(Boolean, default=False)


class ManualCorrection(Base):
    """
    Una correzione manuale applicata su un risultato di segmentazione.
    Ogni correzione è la differenza tra maschera automatica e maschera corretta.
    """
    __tablename__ = "manual_corrections"

    id          = Column(Integer, primary_key=True)
    run_id      = Column(Integer, nullable=False)   # FK logica a SegmentationRun
    patient_id  = Column(String(64), nullable=False, index=True)
    model_name  = Column(String(32))
    cluster_label = Column(Integer)                 # cluster corretto
    correction_type = Column(String(16))            # 'add' | 'remove'
    n_voxels_changed = Column(Integer)
    corrected_mask_path = Column(String(256))
    created_at  = Column(DateTime, default=datetime.utcnow)
    notes       = Column(Text, default="")


class SessionDB:
    """Interfaccia principale al database."""

    def __init__(self, db_path: Path):
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        log.info(f"Database sessioni: {db_path}")

    def save_run(
        self,
        patient_id: str,
        model_name: str,
        n_clusters: int,
        params: dict,
        metrics: dict,
        output_dir: str,
    ) -> int:
        """Salva un'esecuzione e restituisce l'ID."""
        with Session(self.engine) as session:
            run = SegmentationRun(
                patient_id=patient_id,
                model_name=model_name,
                n_clusters=n_clusters,
                params=json.dumps(params),
                metrics=json.dumps(metrics),
                output_dir=str(output_dir),
            )
            session.add(run)
            session.commit()
            run_id = run.id
        log.debug(f"Salvato run #{run_id} ({patient_id}, {model_name})")
        return run_id

    def save_correction(
        self,
        run_id: int,
        patient_id: str,
        model_name: str,
        cluster_label: int,
        correction_type: str,
        n_voxels_changed: int,
        corrected_mask_path: str,
        notes: str = "",
    ) -> None:
        """Salva una correzione manuale."""
        with Session(self.engine) as session:
            corr = ManualCorrection(
                run_id=run_id,
                patient_id=patient_id,
                model_name=model_name,
                cluster_label=cluster_label,
                correction_type=correction_type,
                n_voxels_changed=n_voxels_changed,
                corrected_mask_path=str(corrected_mask_path),
                notes=notes,
            )
            session.add(corr)
            session.commit()
        log.info(
            f"Correzione salvata: {patient_id} run#{run_id} "
            f"cluster={cluster_label} {correction_type} ({n_voxels_changed} voxel)"
        )

    def get_corrections_for_model(self, model_name: str) -> list[dict]:
        """Restituisce tutte le correzioni per un modello — per il fine-tuning."""
        with Session(self.engine) as session:
            rows = session.query(ManualCorrection).filter_by(
                model_name=model_name
            ).all()
        return [
            {
                "patient_id":        r.patient_id,
                "cluster_label":     r.cluster_label,
                "correction_type":   r.correction_type,
                "n_voxels_changed":  r.n_voxels_changed,
                "corrected_mask_path": r.corrected_mask_path,
                "created_at":        r.created_at.isoformat(),
            }
            for r in rows
        ]

    def mark_validated(self, run_id: int) -> None:
        """Segna un run come validato clinicamente."""
        with Session(self.engine) as session:
            run = session.get(SegmentationRun, run_id)
            if run:
                run.is_validated = True
                session.commit()

    def get_patient_history(self, patient_id: str) -> list[dict]:
        """Tutti i run per un paziente, ordinati per data."""
        with Session(self.engine) as session:
            rows = session.query(SegmentationRun).filter_by(
                patient_id=patient_id
            ).order_by(SegmentationRun.created_at.desc()).all()
        return [
            {
                "id":          r.id,
                "model_name":  r.model_name,
                "n_clusters":  r.n_clusters,
                "metrics":     json.loads(r.metrics),
                "created_at":  r.created_at.isoformat(),
                "is_validated": r.is_validated,
            }
            for r in rows
        ]
