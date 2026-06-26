"""
ui/validation_panel.py — Pannello validazione k (v3, multi-modale).

Compatibile con PatientData del refactor. Avvolto in QScrollArea.
Costruisce la FeatureSet dalla modalità e valida k con pesi adattivi.
"""
from __future__ import annotations
import logging
import threading

import numpy as np
from qtpy.QtCore import Qt, Signal, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox,
    QGroupBox, QProgressBar, QTextEdit,
    QScrollArea, QComboBox, QSizePolicy,
)

log = logging.getLogger(__name__)


class _Sig(QObject):
    done  = Signal(object)
    error = Signal(str)


class ValidationPanel(QWidget):
    """
    Pannello di validazione k. Riceve una PatientData via set_patient_data()
    e costruisce la FeatureSet per la modalità corrente.

    Emette k_selected(int) quando l'utente conferma k manualmente.
    """
    k_selected = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = None        # PatientData
        self._report = None
        self._setup_ui()

    def set_patient_data(self, data):
        """Riceve la PatientData dal widget principale."""
        self._data = data
        self.btn_run.setEnabled(True)
        # Popola il selettore feature primaria
        self.cb_primary.clear()
        self.cb_primary.addItems(data.available_channels)
        from io_data.modality import DEFAULT_PRIMARY_FEATURE
        default = DEFAULT_PRIMARY_FEATURE.get(data.modality, data.available_channels[0])
        if default in data.available_channels:
            self.cb_primary.setCurrentText(default)
        self.lbl_status.setText(
            f"Patient: {data.patient_id} [{data.modality.value}] — "
            f"{data.n_tumour_voxels:,} voxel, canali {data.available_channels}"
        )

    # Compatibilità con codice che chiama ancora set_volumes
    def set_volumes(self, data):
        self.set_patient_data(data)

    def _setup_ui(self):
        # Layout esterno con scroll
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)

        self.lbl_status = QLabel("Load a patient in the Validate k tab.")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.lbl_status)

        # Feature primaria
        grp_feat = QGroupBox("Feature to validate")
        hf = QHBoxLayout(grp_feat)
        hf.addWidget(QLabel("Validate on channel:"))
        self.cb_primary = QComboBox()
        hf.addWidget(self.cb_primary)
        layout.addWidget(grp_feat)

        # Parametri
        grp = QGroupBox("Validation parameters")
        hp = QHBoxLayout(grp)
        hp.addWidget(QLabel("k min:"))
        self.sp_k_min = QSpinBox(); self.sp_k_min.setRange(2,4); self.sp_k_min.setValue(2)
        hp.addWidget(self.sp_k_min)
        hp.addWidget(QLabel("k max:"))
        self.sp_k_max = QSpinBox(); self.sp_k_max.setRange(3,8); self.sp_k_max.setValue(6)
        hp.addWidget(self.sp_k_max)
        hp.addWidget(QLabel("Bootstrap:"))
        self.sp_boot = QSpinBox(); self.sp_boot.setRange(0,50); self.sp_boot.setValue(10)
        hp.addWidget(self.sp_boot)
        layout.addWidget(grp)

        self.btn_run = QPushButton("📊 Run validation (criteria analysis for k)")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.hide()
        layout.addWidget(self.progress)

        grp_r = QGroupBox("Results and recommendation")
        vr = QVBoxLayout(grp_r)
        self.txt_report = QTextEdit(); self.txt_report.setReadOnly(True)
        self.txt_report.setMinimumHeight(180)
        self.txt_report.setPlaceholderText("Validation results will appear here.")
        vr.addWidget(self.txt_report)
        layout.addWidget(grp_r)

        self.btn_plots = QPushButton("📈 Show diagnostic plots")
        self.btn_plots.setEnabled(False)
        self.btn_plots.clicked.connect(self._show_plots)
        layout.addWidget(self.btn_plots)

        grp_m = QGroupBox("Manual k selection")
        hm = QHBoxLayout(grp_m)
        hm.addWidget(QLabel("Use k ="))
        self.sp_k_manual = QSpinBox(); self.sp_k_manual.setRange(2,8); self.sp_k_manual.setValue(3)
        hm.addWidget(self.sp_k_manual)
        self.btn_use_k = QPushButton("✔ Use this k for segmentation")
        self.btn_use_k.clicked.connect(lambda: self.k_selected.emit(self.sp_k_manual.value()))
        hm.addWidget(self.btn_use_k)
        layout.addWidget(grp_m)

        nota = QLabel(
            "💡 The recommended k is set automatically, "
            "but you can override it manually."
        )
        nota.setWordWrap(True)
        nota.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(nota)
        layout.addStretch()

    def _run(self):
        if self._data is None:
            return
        self.btn_run.setEnabled(False)
        self.progress.show()
        self.txt_report.setText("⏳ Validation in progress...")

        sig = _Sig()
        sig.done.connect(self._on_done)
        sig.error.connect(self._on_error)

        data    = self._data
        primary = self.cb_primary.currentText()
        k_min   = self.sp_k_min.value()
        k_max   = self.sp_k_max.value()
        n_boot  = self.sp_boot.value()

        def worker():
            try:
                from segmentation.validation import ClusterValidator
                fs = data.build_features(primary_channel=primary, normalize=True)
                validator = ClusterValidator(
                    k_min=k_min, k_max=k_max, n_bootstrap=n_boot,
                )
                report = validator.validate(fs.matrix, data.modality)
                sig.done.emit(report)
            except Exception as e:
                import traceback; traceback.print_exc()
                sig.error.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_done(self, report):
        self._report = report
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.btn_plots.setEnabled(True)
        self.sp_k_manual.setValue(report.k_recommended)
        self.txt_report.setText(report.summary())

    def _on_error(self, msg):
        self.progress.hide()
        self.btn_run.setEnabled(True)
        self.txt_report.setText(f"Error: {msg}")

    def _show_plots(self):
        if self._report is None:
            return
        try:
            self._plot(self._report)
        except Exception as e:
            self.txt_report.append(f"\n[Plot error: {e}]")

    def _plot(self, report):
        import matplotlib.pyplot as plt
        results = report.results
        ks = [r.k for r in results]
        k_rec = report.k_recommended

        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle(f"k diagnostics [{report.modality}] — "
                     f"recommended k = {k_rec}", fontsize=13, fontweight="bold")

        colors = ["#E74C3C" if r.k == k_rec else "#3498DB" for r in results]

        ax = axes[0,0]
        ax.plot(ks, [r.bic for r in results], "b-o", label="BIC")
        ax.plot(ks, [r.aic for r in results], "g--o", label="AIC")
        ax.axvline(k_rec, color="red", lw=2, alpha=0.6)
        ax.set_title("BIC / AIC (min)"); ax.set_xlabel("k"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0,1]
        ax.bar(ks, [r.silhouette for r in results], color=colors)
        ax.axhline(0.3, color="orange", ls="--", alpha=0.5)
        ax.set_title("Silhouette (max)"); ax.set_xlabel("k"); ax.grid(alpha=0.3, axis="y")

        ax = axes[0,2]
        ax.plot(ks, [r.calinski_harabasz for r in results], "purple", marker="o")
        ax.axvline(k_rec, color="red", lw=2, alpha=0.6)
        ax.set_title("Calinski-Harabasz (max)"); ax.set_xlabel("k"); ax.grid(alpha=0.3)

        ax = axes[1,0]
        ax.plot(ks, [r.davies_bouldin for r in results], "darkorange", marker="o")
        ax.axhline(1.0, color="gray", ls="--", alpha=0.4)
        ax.axvline(k_rec, color="red", lw=2, alpha=0.6)
        ax.set_title("Davies-Bouldin (min)"); ax.set_xlabel("k"); ax.grid(alpha=0.3)

        ax = axes[1,1]
        ax.plot(ks, [r.inertia for r in results], "teal", marker="o")
        ax.axvline(k_rec, color="red", lw=2, alpha=0.6)
        ax.set_title("Inertia (Elbow)"); ax.set_xlabel("k"); ax.grid(alpha=0.3)

        ax = axes[1,2]
        stab = [r.stability for r in results]
        if any(stab):
            ax.bar(ks, stab, color=colors)
            ax.axhline(0.8, color="green", ls="--", alpha=0.5)
            ax.set_title("Bootstrap stability (ARI)")
        else:
            ax.text(0.5, 0.5, "Bootstrap disabled", ha="center", va="center")
            ax.set_title("Stability")
        ax.set_xlabel("k"); ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()
