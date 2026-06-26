"""
ui/validation_brats_panel.py — Tab validazione BraTS (v3, refactor multi-modale).

Allineato al nuovo BraTSValidator: usa compute_subregions, niente più mode/add_coords.
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path

from qtpy.QtCore import Signal, QObject, Qt
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QComboBox, QGroupBox, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QCheckBox, QScrollArea,
)

from config.settings import OUTPUT_DIR

log = logging.getLogger(__name__)


class _Sig(QObject):
    done     = Signal(object)
    error    = Signal(str)
    progress = Signal(int, int, str)


class BraTSValidationPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._brats_dir = None
        self._excel_path = None
        self._setup_ui()

    def _setup_ui(self):
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

        grp_data = QGroupBox("BraTS dataset")
        vd = QVBoxLayout(grp_data)
        h = QHBoxLayout()
        self.lbl_brats_path = QLabel("No folder selected")
        self.lbl_brats_path.setStyleSheet("color: gray; font-size: 11px;")
        btn = QPushButton("📁 Select BraTS folder")
        btn.clicked.connect(self._browse)
        h.addWidget(self.lbl_brats_path); h.addWidget(btn)
        vd.addLayout(h)
        hint = QLabel("Subfolders with *-t1n *-t1c *-t2w *-t2f *-seg .nii.gz")
        hint.setStyleSheet("color: gray; font-size: 10px;")
        vd.addWidget(hint)
        layout.addWidget(grp_data)

        grp_p = QGroupBox("Parameters")
        vp = QVBoxLayout(grp_p)
        hm = QHBoxLayout(); hm.addWidget(QLabel("Model:"))
        self.cb_model = QComboBox()
        # Carica dinamicamente i modelli dal registry (così vede anche i nuovi)
        try:
            from io_data.modality import Modality
            from segmentation.registry import list_models_for_modality
            models = list_models_for_modality(Modality.MRI)
        except Exception:
            models = ["GMM", "FCM", "MRF-EM", "DBSCAN", "LevelSet", "Hierarchical"]
        self.cb_model.addItems(models)
        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        hm.addWidget(self.cb_model); vp.addLayout(hm)

        self.grp_gmm = QGroupBox("GMM"); hg = QHBoxLayout(self.grp_gmm)
        hg.addWidget(QLabel("k min:")); self.sp_k_min = QSpinBox()
        self.sp_k_min.setRange(2,4); self.sp_k_min.setValue(2); hg.addWidget(self.sp_k_min)
        hg.addWidget(QLabel("k max:")); self.sp_k_max = QSpinBox()
        self.sp_k_max.setRange(3,6); self.sp_k_max.setValue(4); hg.addWidget(self.sp_k_max)
        vp.addWidget(self.grp_gmm)

        self.grp_k = QGroupBox("k"); hk = QHBoxLayout(self.grp_k)
        hk.addWidget(QLabel("k:")); self.sp_k = QSpinBox()
        self.sp_k.setRange(2,6); self.sp_k.setValue(3); hk.addWidget(self.sp_k)
        self.grp_k.hide(); vp.addWidget(self.grp_k)

        # Parametri Hierarchical
        self.grp_hier = QGroupBox("Hierarchical"); hh = QHBoxLayout(self.grp_hier)
        hh.addWidget(QLabel("Primary weight:"))
        self.sp_pw = QDoubleSpinBox(); self.sp_pw.setRange(1.0, 4.0)
        self.sp_pw.setValue(2.0); self.sp_pw.setSingleStep(0.5); hh.addWidget(self.sp_pw)
        hh.addWidget(QLabel("Macro-regions:"))
        self.sp_l1 = QSpinBox(); self.sp_l1.setRange(2,3); self.sp_l1.setValue(2)
        hh.addWidget(self.sp_l1)
        self.grp_hier.hide(); vp.addWidget(self.grp_hier)

        hc = QHBoxLayout(); hc.addWidget(QLabel("Max cases (0=all):"))
        self.sp_max = QSpinBox(); self.sp_max.setRange(0,1000); self.sp_max.setValue(10)
        hc.addWidget(self.sp_max); vp.addLayout(hc)

        self.chk_subregions = QCheckBox(
            "Compute TC/ET subregions (a-priori mapping)"
        )
        vp.addWidget(self.chk_subregions)
        layout.addWidget(grp_p)

        nota = QLabel(
            "ℹ Real MRI mode: features [T1, T1ce, T2, FLAIR].\n"
            "Metrics: Dice, Jaccard, HD95, ARI, runtime.\n"
            "Cluster→region mapping fixed a priori by intensity."
        )
        nota.setWordWrap(True)
        nota.setStyleSheet("color: #1A3A5C; background: #EAF4FB; "
                           "border-radius: 4px; padding: 8px; font-size: 10px;")
        layout.addWidget(nota)

        self.btn_run = QPushButton("▶  Start BraTS validation")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)
        self.progress = QProgressBar(); self.progress.setRange(0,100); self.progress.hide()
        layout.addWidget(self.progress)
        self.lbl_prog = QLabel(""); self.lbl_prog.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.lbl_prog)

        grp_r = QGroupBox("Results"); vr = QVBoxLayout(grp_r)
        self.txt_results = QTextEdit(); self.txt_results.setReadOnly(True)
        self.txt_results.setPlaceholderText(
            "Feature MRI: T1, T1ce, T2, FLAIR\n"
            "Dice WT/TC/ET + Jaccard + HD95 + ARI + runtime"
        )
        vr.addWidget(self.txt_results)
        self.btn_excel = QPushButton("📊 Open Excel")
        self.btn_excel.setEnabled(False); self.btn_excel.clicked.connect(self._open_excel)
        vr.addWidget(self.btn_excel)
        layout.addWidget(grp_r)

    def _on_model_changed(self, name):
        self.grp_gmm.setVisible(name == "GMM")
        self.grp_hier.setVisible(name == "Hierarchical")
        self.grp_k.setVisible(name not in ("GMM", "Hierarchical"))

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(self, "BraTS folder")
        if folder:
            self._brats_dir = Path(folder)
            cases = [d for d in self._brats_dir.iterdir() if d.is_dir()]
            self.lbl_brats_path.setText(f"{folder[-50:]} ({len(cases)} cases)")
            self.lbl_brats_path.setStyleSheet("color: #1E8449; font-size: 11px;")
            self.btn_run.setEnabled(True)

    def _params(self):
        m = self.cb_model.currentText()
        if m == "GMM":
            return {"k_min": self.sp_k_min.value(), "k_max": self.sp_k_max.value()}
        elif m == "Hierarchical":
            return {"primary_weight": self.sp_pw.value(), "n_level1": self.sp_l1.value()}
        elif m == "FCM":
            return {"k": self.sp_k.value()}
        elif m == "MRF-EM":
            return {"k": self.sp_k.value(), "beta": 1.5, "max_iter": 10}
        elif m == "LevelSet":
            return {"n_clusters": self.sp_k.value()}
        return {}

    def _run(self):
        if self._brats_dir is None:
            return
        model = self.cb_model.currentText()
        params = self._params()
        max_cases = self.sp_max.value() or None
        subregions = self.chk_subregions.isChecked()

        self.btn_run.setEnabled(False); self.progress.show(); self.progress.setValue(0)
        self.txt_results.setText("⏳ Validation in progress...")
        sig = _Sig()
        sig.done.connect(self._on_done)
        sig.error.connect(self._on_error)
        sig.progress.connect(self._on_progress)
        out_dir = OUTPUT_DIR / "brats_validation"

        def worker():
            try:
                from validation.brats_benchmark import BraTSValidator
                v = BraTSValidator(
                    brats_dir=self._brats_dir, output_dir=out_dir,
                    model_name=model, model_params=params,
                    max_cases=max_cases, compute_subregions=subregions,
                )
                report = v.run(progress_callback=lambda c,t,p: sig.progress.emit(c,t,p))
                sig.done.emit(report)
            except Exception as e:
                import traceback; traceback.print_exc()
                sig.error.emit(str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _on_progress(self, c, t, pid):
        self.progress.setValue(int((c+1)/max(t,1)*100))
        self.lbl_prog.setText(f"Case {c+1}/{t}: {pid}")

    def _on_done(self, report):
        self.progress.hide(); self.btn_run.setEnabled(True)
        self.lbl_prog.setText(f"✔ {report.n_cases-report.n_failed}/{report.n_cases} cases")
        self.txt_results.setText(report.summary())
        self._excel_path = OUTPUT_DIR / "brats_validation" / f"brats_MRI_{report.model_name}.xlsx"
        self.btn_excel.setEnabled(True)

    def _on_error(self, msg):
        self.progress.hide(); self.btn_run.setEnabled(True); self.lbl_prog.setText("")
        QMessageBox.critical(self, "BraTS error", msg)

    def _open_excel(self):
        if self._excel_path and Path(self._excel_path).exists():
            import subprocess, sys
            if sys.platform == "win32":
                subprocess.run(["start", str(self._excel_path)], shell=True)
            else:
                subprocess.run(["xdg-open", str(self._excel_path)])
