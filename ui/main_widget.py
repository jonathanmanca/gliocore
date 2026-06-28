"""
ui/main_widget.py — GlioCore pannello principale (v5, multi-modale).

Aggiornamenti per il refactor:
- Usa PatientData + FeatureSet + SegmentationContext (nuova firma fit)
- Rileva automaticamente la modalità (PET / MRI / PET_MRI)
- Selettore feature primaria configurabile per il cluster ordering
- Mostra la modalità rilevata e i canali disponibili
- Filtra i modelli compatibili con la modalità corrente
"""
from __future__ import annotations
import logging
import threading
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, Signal, QObject
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel,
    QSpinBox, QDoubleSpinBox, QTextEdit,
    QGroupBox, QProgressBar, QCheckBox,
    QFileDialog, QMessageBox, QTabWidget,
    QSizePolicy, QSlider, QScrollArea,
)

from config.settings import DATA_DIR, OUTPUT_DIR, DB_PATH, OVERLAY_OPACITY
from io_data.loader import load_patient, save_nifti_like, PatientData
from io_data.modality import Modality
from segmentation.registry import (
    get_model, list_models, list_models_for_modality,
)
from agents.research_agent import ResearchAgent, ModelAgent
from learning.session_db import SessionDB

log = logging.getLogger(__name__)


class _Signals(QObject):
    finished = Signal(object)
    error    = Signal(str)


def _set_label_color(layer, label_id, color_name):
    """Imposta colore label in napari 0.7."""
    try:
        from napari.utils.colormaps import DirectLabelColormap
        import matplotlib.colors as mcolors
        rgba = mcolors.to_rgba(color_name)
        layer.colormap = DirectLabelColormap(
            color_dict={label_id: rgba, None: (0, 0, 0, 0)}
        )
    except Exception as e:
        log.debug(f"Color not set: {e}")


class GlioCoreWidget(QWidget):

    def __init__(self, napari_viewer, parent=None):
        super().__init__(parent)
        self.viewer = napari_viewer
        self.db = SessionDB(DB_PATH)
        self.data: PatientData | None = None
        self.result = None
        self.run_id = None
        self._agents = {"research": ResearchAgent(), "model": ModelAgent()}
        self._correction_active = False
        self._original_labels = None
        self._atlas_report = None
        self._mask_mni_path = None
        self._t1_path = None
        self._custom_dirs = {}
        self._tabs = None
        self._val_panel = None
        self._setup_ui()
        self._refresh_patient_list()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        # Barra About
        h_top = QHBoxLayout()
        h_top.addStretch()
        try:
            from ui.about_dialog import AboutDialog
            btn_about = QPushButton("ℹ  About GlioCore")
            btn_about.setFixedHeight(22)
            btn_about.setStyleSheet(
                "color: #1A3A5C; border: none; font-size: 11px; "
                "font-weight: bold; background: transparent;"
            )
            btn_about.clicked.connect(lambda: AboutDialog(self).exec_())
            h_top.addWidget(btn_about)
        except ImportError:
            pass
        root.addLayout(h_top)

        self._tabs = QTabWidget()

        # Pannello validazione k
        try:
            from ui.validation_panel import ValidationPanel
            self._val_panel = ValidationPanel()
            if hasattr(self._val_panel, "k_selected"):
                self._val_panel.k_selected.connect(self._apply_validated_k)
        except ImportError:
            self._val_panel = None

        if self._val_panel:
            self._tabs.addTab(self._build_validate_tab(), "📊 Validate k")
        self._tabs.addTab(self._build_segmentation_tab(), "🧠 Segmentation")
        self._tabs.addTab(self._build_correction_tab(),   "✏️ Correction")
        self._tabs.addTab(self._build_atlas_tab(),        "🗺 Atlas WM")

        # BraTS validation
        try:
            from ui.validation_brats_panel import BraTSValidationPanel
            self._brats_panel = BraTSValidationPanel()
            self._tabs.addTab(self._brats_panel, "🔬 Validate BraTS")
        except ImportError:
            pass

        self._tabs.addTab(self._build_agent_tab(),   "🤖 AI Agents")
        self._tabs.addTab(self._build_history_tab(), "📋 History")
        root.addWidget(self._tabs)

    def _build_validate_tab(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)
        grp_p = QGroupBox("Patient")
        vp = QVBoxLayout(grp_p)
        h = QHBoxLayout()
        self.cb_patient = QComboBox()
        self.cb_patient.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_browse = QPushButton("📁"); btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_data_dir)
        h.addWidget(QLabel("Patient:")); h.addWidget(self.cb_patient); h.addWidget(btn_browse)
        vp.addLayout(h)
        btn_load = QPushButton("Load volumes")
        btn_load.clicked.connect(self._load_patient)
        vp.addWidget(btn_load)
        self.lbl_info = QLabel("No patient loaded")
        self.lbl_info.setStyleSheet("color: gray; font-size: 11px;")
        vp.addWidget(self.lbl_info)
        layout.addWidget(grp_p)

        hint = QLabel("Load the patient, then validate k and go to Segmentation.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #1565C0; font-size: 11px; padding: 2px;")
        layout.addWidget(hint)
        if self._val_panel:
            layout.addWidget(self._val_panel)
        return outer

    def _build_segmentation_tab(self):
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)
        w = QWidget()
        scroll.setWidget(w)
        layout = QVBoxLayout(w)

        # Info modalità
        self.lbl_modality = QLabel("Load a patient first in the 📊 Validate k tab")
        self.lbl_modality.setStyleSheet(
            "color: #1A3A5C; background: #EAF4FB; border-radius: 4px; "
            "padding: 6px; font-size: 11px;"
        )
        self.lbl_modality.setWordWrap(True)
        layout.addWidget(self.lbl_modality)

        # Feature primaria (cluster ordering)
        grp_feat = QGroupBox("Feature for cluster ordering")
        vf = QVBoxLayout(grp_feat)
        h_feat = QHBoxLayout()
        h_feat.addWidget(QLabel("Order clusters by:"))
        self.cb_primary = QComboBox()
        self.cb_primary.setEnabled(False)
        h_feat.addWidget(self.cb_primary)
        h_feat.addWidget(QLabel("Order:"))
        self.cb_order = QComboBox()
        self.cb_order.addItems(["ascending (1=low)", "descending (1=high)"])
        h_feat.addWidget(self.cb_order)
        vf.addLayout(h_feat)
        nota_feat = QLabel(
            "Cluster 1 = lowest value of the chosen feature (e.g. low SUVR = necrosis). "
            "Cluster k = highest value (infiltration)."
        )
        nota_feat.setWordWrap(True)
        nota_feat.setStyleSheet("color: gray; font-size: 10px;")
        vf.addWidget(nota_feat)
        layout.addWidget(grp_feat)

        # Modello
        grp_m = QGroupBox("Model")
        vm = QVBoxLayout(grp_m)
        self.cb_model = QComboBox()
        self.cb_model.currentTextChanged.connect(self._on_model_changed)
        vm.addWidget(self.cb_model)
        self.lbl_model_desc = QLabel()
        self.lbl_model_desc.setWordWrap(True)
        self.lbl_model_desc.setStyleSheet("color: gray; font-size: 10px;")
        vm.addWidget(self.lbl_model_desc)

        self.grp_gmm = self._params_gmm()
        self.grp_fcm = self._params_fcm()
        self.grp_ls  = self._params_ls()
        self.grp_mrf = self._params_mrf()
        self.grp_dbs = self._params_dbs()
        for g in [self.grp_gmm, self.grp_fcm, self.grp_ls, self.grp_mrf, self.grp_dbs]:
            vm.addWidget(g)
        layout.addWidget(grp_m)

        self.btn_run = QPushButton("▶  Run segmentation")
        self.btn_run.setEnabled(False)
        self.btn_run.setStyleSheet("font-weight: bold; padding: 6px;")
        self.btn_run.clicked.connect(self._run_segmentation)
        layout.addWidget(self.btn_run)
        self.progress = QProgressBar(); self.progress.setRange(0,0); self.progress.hide()
        layout.addWidget(self.progress)

        grp_r = QGroupBox("Results")
        vr = QVBoxLayout(grp_r)
        self.txt_results = QTextEdit(); self.txt_results.setReadOnly(True)
        self.txt_results.setMaximumHeight(130)
        vr.addWidget(self.txt_results)
        h2 = QHBoxLayout()
        self.btn_validate_run = QPushButton("✔ Validate"); self.btn_validate_run.setEnabled(False)
        self.btn_validate_run.clicked.connect(self._validate_result)
        self.btn_save = QPushButton("💾 Save NIfTI"); self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self._save_result)
        h2.addWidget(self.btn_validate_run); h2.addWidget(self.btn_save)
        vr.addLayout(h2)
        layout.addWidget(grp_r)
        layout.addStretch()
        return outer

    def _params_gmm(self):
        g = QGroupBox("GMM parameters"); h = QHBoxLayout(g)
        h.addWidget(QLabel("k min:")); self.sp_k_min = QSpinBox()
        self.sp_k_min.setRange(2,8); self.sp_k_min.setValue(2); h.addWidget(self.sp_k_min)
        h.addWidget(QLabel("k max:")); self.sp_k_max = QSpinBox()
        self.sp_k_max.setRange(2,8); self.sp_k_max.setValue(6); h.addWidget(self.sp_k_max)
        h.addWidget(QLabel("Criterion:")); self.cb_criterion = QComboBox()
        self.cb_criterion.addItems(["BIC","AIC"]); h.addWidget(self.cb_criterion)
        return g

    def _params_fcm(self):
        g = QGroupBox("FCM parameters"); h = QHBoxLayout(g)
        h.addWidget(QLabel("k:")); self.sp_fcm_k = QSpinBox()
        self.sp_fcm_k.setRange(2,8); self.sp_fcm_k.setValue(3); h.addWidget(self.sp_fcm_k)
        self.chk_auto_k = QCheckBox("Auto k"); h.addWidget(self.chk_auto_k)
        h.addWidget(QLabel("m:")); self.sp_fcm_m = QDoubleSpinBox()
        self.sp_fcm_m.setRange(1.1,3.0); self.sp_fcm_m.setValue(2.0)
        self.sp_fcm_m.setSingleStep(0.1); h.addWidget(self.sp_fcm_m)
        return g

    def _params_ls(self):
        g = QGroupBox("Level Set parameters"); h = QHBoxLayout(g)
        h.addWidget(QLabel("Iter:")); self.sp_ls_iter = QSpinBox()
        self.sp_ls_iter.setRange(50,500); self.sp_ls_iter.setValue(200)
        self.sp_ls_iter.setSingleStep(50); h.addWidget(self.sp_ls_iter)
        h.addWidget(QLabel("Clusters:")); self.sp_ls_k = QSpinBox()
        self.sp_ls_k.setRange(2,3); self.sp_ls_k.setValue(2); h.addWidget(self.sp_ls_k)
        return g

    def _params_mrf(self):
        g = QGroupBox("MRF-EM parameters"); h = QHBoxLayout(g)
        h.addWidget(QLabel("β:")); self.sp_beta = QDoubleSpinBox()
        self.sp_beta.setRange(0.0,5.0); self.sp_beta.setValue(1.5)
        self.sp_beta.setSingleStep(0.5); h.addWidget(self.sp_beta)
        h.addWidget(QLabel("k:")); self.sp_mrf_k = QSpinBox()
        self.sp_mrf_k.setRange(2,6); self.sp_mrf_k.setValue(3); h.addWidget(self.sp_mrf_k)
        h.addWidget(QLabel("Max iter:")); self.sp_mrf_iter = QSpinBox()
        self.sp_mrf_iter.setRange(5,50); self.sp_mrf_iter.setValue(20); h.addWidget(self.sp_mrf_iter)
        return g

    def _params_dbs(self):
        g = QGroupBox("DBSCAN parameters"); h = QHBoxLayout(g)
        h.addWidget(QLabel("Spatial weight:")); self.sp_spatial_w = QDoubleSpinBox()
        self.sp_spatial_w.setRange(0.05,1.0); self.sp_spatial_w.setValue(0.3)
        self.sp_spatial_w.setSingleStep(0.05); h.addWidget(self.sp_spatial_w)
        return g

    def _build_correction_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        grp = QGroupBox("Instructions"); vi = QVBoxLayout(grp)
        vi.addWidget(QLabel(
            "1. Run a segmentation\n"
            "2. Click 'Enable correction mode'\n"
            "3. Brush (key 2), eraser (key 1), [ ] brush size\n"
            "4. Save correction"
        ))
        layout.addWidget(grp)
        grp_c = QGroupBox("Controls"); vc = QVBoxLayout(grp_c)
        self.btn_activate_correction = QPushButton("🖊 Enable correction mode")
        self.btn_activate_correction.setEnabled(False)
        self.btn_activate_correction.clicked.connect(self._activate_correction_mode)
        vc.addWidget(self.btn_activate_correction)
        h_op = QHBoxLayout(); h_op.addWidget(QLabel("Opacity:"))
        self.sld_opacity = QSlider(Qt.Horizontal)
        self.sld_opacity.setRange(0,100); self.sld_opacity.setValue(int(OVERLAY_OPACITY*100))
        self.sld_opacity.valueChanged.connect(self._update_opacity)
        h_op.addWidget(self.sld_opacity)
        self.lbl_opacity_val = QLabel(f"{int(OVERLAY_OPACITY*100)}%")
        h_op.addWidget(self.lbl_opacity_val); vc.addLayout(h_op)
        vc.addWidget(QLabel("Clinical notes:"))
        self.txt_correction_notes = QTextEdit(); self.txt_correction_notes.setMaximumHeight(60)
        vc.addWidget(self.txt_correction_notes)
        h_b = QHBoxLayout()
        self.btn_save_correction = QPushButton("💾 Save")
        self.btn_save_correction.setEnabled(False)
        self.btn_save_correction.clicked.connect(self._save_correction)
        self.btn_undo_correction = QPushButton("↩ Undo")
        self.btn_undo_correction.setEnabled(False)
        self.btn_undo_correction.clicked.connect(self._undo_correction)
        h_b.addWidget(self.btn_save_correction); h_b.addWidget(self.btn_undo_correction)
        vc.addLayout(h_b); layout.addWidget(grp_c)
        self.lbl_correction_count = QLabel("Corrections: 0")
        layout.addWidget(self.lbl_correction_count)
        layout.addStretch()
        return w

    def _build_atlas_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        grp = QGroupBox("MNI152 registration"); vr = QVBoxLayout(grp)
        h = QHBoxLayout()
        self.lbl_t1_path = QLabel("Automatic T1")
        self.lbl_t1_path.setStyleSheet("color: gray; font-size: 10px;")
        btn_t1 = QPushButton("📁 T1"); btn_t1.clicked.connect(self._browse_t1)
        h.addWidget(self.lbl_t1_path); h.addWidget(btn_t1); vr.addLayout(h)
        ht = QHBoxLayout(); ht.addWidget(QLabel("Type:"))
        self.cb_reg_type = QComboBox()
        self.cb_reg_type.addItems(["Affine (~1min)", "SyN (~5min)"])
        ht.addWidget(self.cb_reg_type); vr.addLayout(ht)
        self.btn_register = QPushButton("🧭 Register MNI")
        self.btn_register.setEnabled(False)
        self.btn_register.clicked.connect(self._run_registration)
        vr.addWidget(self.btn_register)
        self.prog_atlas = QProgressBar(); self.prog_atlas.setRange(0,0); self.prog_atlas.hide()
        vr.addWidget(self.prog_atlas); layout.addWidget(grp)
        grp_a = QGroupBox("WM tract analysis"); va = QVBoxLayout(grp_a)
        hm = QHBoxLayout(); hm.addWidget(QLabel("Analyze:"))
        self.cb_atlas_mask = QComboBox()
        self.cb_atlas_mask.addItems(["Whole tumor", "Hyper only", "Hypo only"])
        hm.addWidget(self.cb_atlas_mask); va.addLayout(hm)
        self.btn_analyze_atlas = QPushButton("📊 Compute overlap")
        self.btn_analyze_atlas.setEnabled(False)
        self.btn_analyze_atlas.clicked.connect(self._run_atlas_analysis)
        va.addWidget(self.btn_analyze_atlas)
        self.txt_atlas_results = QTextEdit(); self.txt_atlas_results.setReadOnly(True)
        va.addWidget(self.txt_atlas_results)
        self.btn_save_atlas_csv = QPushButton("📄 Save CSV")
        self.btn_save_atlas_csv.setEnabled(False)
        self.btn_save_atlas_csv.clicked.connect(self._save_atlas_csv)
        va.addWidget(self.btn_save_atlas_csv); layout.addWidget(grp_a)
        layout.addStretch()
        return w

    def _build_agent_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        grp = QGroupBox("Actions"); va = QVBoxLayout(grp)
        for label, fn in [
            ("🔍 Suggest methods", self._agent_suggest_methods),
            ("📊 Interpret result", self._agent_interpret),
            ("📄 Generate clinical summary", self._agent_report),
        ]:
            b = QPushButton(label); b.clicked.connect(fn); va.addWidget(b)
        layout.addWidget(grp)
        self.txt_agent = QTextEdit(); self.txt_agent.setReadOnly(True)
        layout.addWidget(self.txt_agent)
        return w

    def _build_history_tab(self):
        w = QWidget(); layout = QVBoxLayout(w)
        btn = QPushButton("🔄 Refresh"); btn.clicked.connect(self._refresh_history)
        layout.addWidget(btn)
        self.txt_history = QTextEdit(); self.txt_history.setReadOnly(True)
        layout.addWidget(self.txt_history)
        return w

    # ── Caricamento ──────────────────────────────────────────────────────────

    def _refresh_patient_list(self):
        self.cb_patient.clear()
        if DATA_DIR.exists():
            self.cb_patient.addItems(
                sorted([p.name for p in DATA_DIR.iterdir() if p.is_dir()])
            )

    def _browse_data_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder")
        if folder:
            folder = Path(folder)
            self._custom_dirs[folder.name] = folder
            self.cb_patient.addItem(folder.name)
            self.cb_patient.setCurrentText(folder.name)

    def _load_patient(self):
        name = self.cb_patient.currentText()
        if not name:
            return
        patient_dir = self._custom_dirs.get(name, DATA_DIR / name)
        try:
            self.data = load_patient(patient_dir)
        except Exception as e:
            QMessageBox.critical(self, "Loading error", str(e)); return

        modality = self.data.modality
        channels = self.data.available_channels
        info = (f"✔ {name} [{modality.value}] — "
                f"{self.data.n_tumour_voxels:,} voxels | channels: {channels}")
        self.lbl_info.setText(info)
        self.lbl_modality.setText(
            f"Modality: {modality.value}  |  Available channels: {', '.join(channels)}\n"
            f"Tumor voxels: {self.data.n_tumour_voxels:,}  |  Shape: {self.data.shape}"
        )

        # Popola selettore feature primaria
        self.cb_primary.clear()
        self.cb_primary.addItems(channels)
        self.cb_primary.setEnabled(True)
        # Default per modalità
        from io_data.modality import DEFAULT_PRIMARY_FEATURE
        default_ch = DEFAULT_PRIMARY_FEATURE.get(modality, channels[0])
        if default_ch in channels:
            self.cb_primary.setCurrentText(default_ch)

        # Filtra modelli compatibili con la modalità
        compatible = list_models_for_modality(modality)
        self.cb_model.clear()
        self.cb_model.addItems(compatible)
        self._on_model_changed(self.cb_model.currentText())

        self._show_volumes_in_napari()
        self.btn_run.setEnabled(True)
        self.btn_activate_correction.setEnabled(False)
        has_t1 = "t1" in channels
        self.btn_register.setEnabled(has_t1)

        if self._val_panel and hasattr(self._val_panel, "set_volumes"):
            # Il pannello validazione si aspetta un oggetto con .suvr/.mask
            # Passiamo la feature primaria come array
            try:
                self._val_panel.set_patient_data(self.data)
            except AttributeError:
                pass

    def _show_volumes_in_napari(self):
        self.viewer.layers.clear()

        def _enforce_pan_zoom(event=None):
            if not self._correction_active:
                for layer in self.viewer.layers:
                    if hasattr(layer, 'mode') and layer.mode != 'pan_zoom':
                        layer.mode = 'pan_zoom'
        try:
            self.viewer.layers.selection.events.changed.connect(_enforce_pan_zoom)
        except Exception:
            pass

        d = self.data
        voxel_size = tuple(abs(float(d.affine[i, i])) for i in range(3))
        bg, bg_name = d.get_background()
        self.viewer.add_image(bg, name=bg_name, colormap="gray", scale=voxel_size)

        # Mostra anche SUVR se presente (per PET)
        if "suvr" in d.volumes:
            self.viewer.add_image(
                d.volumes["suvr"], name="SUVR", colormap="magma",
                opacity=0.6, blending="additive", scale=voxel_size
            )

        ml = self.viewer.add_labels(
            d.mask.astype(np.uint8), name="Tumor mask",
            opacity=0.35, scale=voxel_size
        )
        ml.mode = "pan_zoom"

    # ── Segmentazione ────────────────────────────────────────────────────────

    def _apply_validated_k(self, k: int):
        self.sp_k_min.setValue(k)
        self.sp_k_max.setValue(k)
        for i in range(self._tabs.count()):
            if "Segmentation" in self._tabs.tabText(i):
                self._tabs.setCurrentIndex(i); break
        QMessageBox.information(self, "k set",
                                f"k={k} set. Click 'Run segmentation'.")

    def _on_model_changed(self, model_name):
        models = {m["name"]: m["description"] for m in list_models()}
        self.lbl_model_desc.setText(models.get(model_name, ""))
        self.grp_gmm.setVisible(model_name == "GMM")
        self.grp_fcm.setVisible(model_name == "FCM")
        self.grp_ls.setVisible(model_name == "LevelSet")
        self.grp_mrf.setVisible(model_name == "MRF-EM")
        self.grp_dbs.setVisible(model_name == "DBSCAN")

    def _collect_params(self, model_name):
        if model_name == "GMM":
            return {"k_min": self.sp_k_min.value(), "k_max": self.sp_k_max.value(),
                    "criterion": self.cb_criterion.currentText()}
        elif model_name == "FCM":
            return {"k": self.sp_fcm_k.value(), "m": self.sp_fcm_m.value(),
                    "auto_k": self.chk_auto_k.isChecked()}
        elif model_name == "LevelSet":
            return {"num_iter": self.sp_ls_iter.value(), "n_clusters": self.sp_ls_k.value()}
        elif model_name == "MRF-EM":
            return {"beta": self.sp_beta.value(), "k": self.sp_mrf_k.value(),
                    "max_iter": self.sp_mrf_iter.value()}
        elif model_name == "DBSCAN":
            return {"spatial_weight": self.sp_spatial_w.value()}
        return {}

    def _run_segmentation(self):
        if self.data is None:
            QMessageBox.warning(self, "No patient", "Load a patient first."); return

        model_name = self.cb_model.currentText()
        params = self._collect_params(model_name)
        primary = self.cb_primary.currentText()
        order = "ascending" if "ascending" in self.cb_order.currentText() else "descending"

        self.btn_run.setEnabled(False); self.progress.show(); self.txt_results.clear()

        sig = _Signals()
        sig.finished.connect(self._on_segmentation_done)
        sig.error.connect(self._on_segmentation_error)
        d = self.data

        def worker():
            try:
                features = d.build_features(primary_channel=primary, normalize=True)
                context = d.build_context(primary_channel=primary, cluster_order=order)
                model = get_model(model_name, **params)
                result = model.fit(features, context)
                sig.finished.emit(result)
            except Exception as e:
                import traceback; traceback.print_exc()
                sig.error.emit(str(e))

        threading.Thread(target=worker, daemon=True).start()

    def _on_segmentation_done(self, result):
        self.result = result
        self.progress.hide(); self.btn_run.setEnabled(True)
        self.btn_validate_run.setEnabled(True); self.btn_save.setEnabled(True)
        self.btn_activate_correction.setEnabled(True)

        lines = [f"✔ Model: {result.model_name} [{result.modality}]",
                 f"  Clusters: {result.n_clusters}",
                 f"  Primary feature: {result.primary_channel}"]
        for k, v in result.metrics.items():
            if k not in ("all_models", "channels"):
                if isinstance(v, float):
                    lines.append(f"  {k}: {v:.4f}")
                elif not isinstance(v, (list, dict)):
                    lines.append(f"  {k}: {v}")
        self.txt_results.setText("\n".join(lines))
        self._show_result_in_napari(result)

        self.run_id = self.db.save_run(
            patient_id=self.data.patient_id, model_name=result.model_name,
            n_clusters=result.n_clusters, params=result.params,
            metrics={k: v for k, v in result.metrics.items()
                     if not isinstance(v, (list, dict))},
            output_dir=str(OUTPUT_DIR / self.data.patient_id),
        )

    def _on_segmentation_error(self, msg):
        self.progress.hide(); self.btn_run.setEnabled(True)
        QMessageBox.critical(self, "Segmentation error", msg)

    def _show_result_in_napari(self, result):
        to_remove = [l for l in self.viewer.layers
                     if any(kw in l.name.lower() for kw in
                            ["cluster","hypo","hyper","intensity"])]
        for l in to_remove:
            self.viewer.layers.remove(l)
        voxel_size = tuple(abs(float(self.data.affine[i,i])) for i in range(3))

        # Nomi maschere dipendenti dalla modalità:
        # PET / PET+MRI → significato metabolico (Hypo/Hyper)
        # MRI → significato di intensità (i cluster sono ordinati per T1ce)
        from io_data.modality import Modality
        if self.data.modality == Modality.MRI:
            low_name, high_name = "Low intensity", "High intensity"
        else:
            low_name, high_name = "Hypo", "Hyper"

        cl = self.viewer.add_labels(result.label_volume,
                                    name=f"Clusters [{result.model_name}]",
                                    opacity=OVERLAY_OPACITY, scale=voxel_size)
        cl.mode = "pan_zoom"
        hl = self.viewer.add_labels(result.hypo_mask.astype(np.uint8),
                                    name=low_name, opacity=0.6, scale=voxel_size)
        hl.mode = "pan_zoom"; _set_label_color(hl, 1, "cornflowerblue")
        yl = self.viewer.add_labels(result.hyper_mask.astype(np.uint8),
                                    name=high_name, opacity=0.6, scale=voxel_size)
        yl.mode = "pan_zoom"; _set_label_color(yl, 1, "tomato")

    # ── Correzione ───────────────────────────────────────────────────────────

    def _activate_correction_mode(self):
        if self.result is None:
            return
        self._correction_active = True
        self._original_labels = self.result.label_volume.copy()
        self.btn_save_correction.setEnabled(True)
        self.btn_undo_correction.setEnabled(True)
        for layer in self.viewer.layers:
            if "Cluster" in layer.name and hasattr(layer, "mode"):
                self.viewer.layers.selection.active = layer
                layer.mode = "paint"; break
        QMessageBox.information(
            self, "Correction mode active",
            "You can now edit the segmentation directly on the image.\n\n"
            "napari brush controls:\n"
            "   •  Draw:  hold and drag the mouse\n"
            "   •  Erase:  select the eraser in the layer controls\n"
            "   •  Brush size:  keys  [  and  ]\n"
            "   •  Undo:  Ctrl+Z\n\n"
            "When done, press 'Save correction' to store the changes."
        )

    def _update_opacity(self, value):
        self.lbl_opacity_val.setText(f"{value}%")
        for layer in self.viewer.layers:
            if any(kw in layer.name for kw in
                   ["Cluster","Hypo","Hyper","intensity"]):
                layer.opacity = value / 100.0

    def _save_correction(self):
        if not self._correction_active or self.result is None:
            return
        corrected = None
        for layer in self.viewer.layers:
            if "Cluster" in layer.name and hasattr(layer, "data"):
                corrected = np.array(layer.data, dtype=np.uint8); break
        if corrected is None:
            return
        n_changed = int((corrected != self._original_labels).sum())
        if n_changed == 0:
            QMessageBox.information(self, "No changes", "Nothing to save."); return
        out_dir = OUTPUT_DIR / self.data.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        corr_path = out_dir / f"clusters_{self.result.model_name}_corrected.nii.gz"
        save_nifti_like(self.data.reference_img, corrected, corr_path)
        self.db.save_correction(
            run_id=self.run_id or 0, patient_id=self.data.patient_id,
            model_name=self.result.model_name, cluster_label=0,
            correction_type="edit", n_voxels_changed=n_changed,
            corrected_mask_path=str(corr_path),
            notes=self.txt_correction_notes.toPlainText())

        # Active learning: salva (feature aumentate, label corretti) per il
        # retraining incrementale del Random Forest. Le feature sono ricostruite
        # con la stessa pipeline usata dall'RF, allineate ai voxel della maschera.
        try:
            from segmentation.bayesian_rf import BayesianRFSegmentation
            primary = self.result.primary_channel
            feats = self.data.build_features(primary_channel=primary, normalize=True)
            ctx = self.data.build_context(primary_channel=primary)
            X = BayesianRFSegmentation()._augment_features(feats, ctx)
            y = corrected[self.data.mask]
            if len(X) == len(y):
                BayesianRFSegmentation.save_training_sample(
                    X, y, self.data.patient_id, tag=self.result.model_name)
        except Exception as e:
            log.warning(f"Active learning: training sample not saved ({e})")

        for layer in self.viewer.layers:
            if "Cluster" in layer.name and hasattr(layer, "mode"):
                layer.mode = "pan_zoom"; break
        self._correction_active = False
        self.btn_save_correction.setEnabled(False)
        self.btn_undo_correction.setEnabled(False)
        QMessageBox.information(self, "Saved", f"✔ {n_changed:,} voxels saved.")

    def _undo_correction(self):
        if self._original_labels is None:
            return
        for layer in self.viewer.layers:
            if "Cluster" in layer.name and hasattr(layer, "data"):
                layer.data = self._original_labels.copy()
                layer.mode = "pan_zoom"; break
        self._correction_active = False
        self.btn_save_correction.setEnabled(False)
        self.btn_undo_correction.setEnabled(False)

    # ── Atlas (usa data.reference_img e data.mask) ───────────────────────────

    def _browse_t1(self):
        path, _ = QFileDialog.getOpenFileName(self, "T1", "", "NIfTI (*.nii *.nii.gz)")
        if path:
            self._t1_path = Path(path)
            self.lbl_t1_path.setText(path[-50:])
            self.btn_register.setEnabled(True)

    def _run_registration(self):
        if self.data is None:
            return
        t1_path = self._t1_path
        if t1_path is None and "t1" in self.data.images:
            t1_path = OUTPUT_DIR / self.data.patient_id / "T1_temp.nii.gz"
            save_nifti_like(self.data.images["t1"], self.data.volumes["t1"],
                            t1_path, dtype=np.float32)
        if t1_path is None:
            QMessageBox.warning(self, "T1 missing", "Select a T1."); return
        mask_path = OUTPUT_DIR / self.data.patient_id / "mask_for_reg.nii.gz"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        save_nifti_like(self.data.reference_img, self.data.mask.astype(np.uint8), mask_path)
        reg_type = "Affine" if "Affine" in self.cb_reg_type.currentText() else "SyN"
        out_dir = OUTPUT_DIR / self.data.patient_id / "mni_registration"
        self.btn_register.setEnabled(False); self.prog_atlas.show()
        self.txt_atlas_results.setText(f"⏳ {reg_type} registration...")
        sig = _Signals()
        sig.finished.connect(self._on_registration_done)
        sig.error.connect(lambda e: (self.prog_atlas.hide(),
                                     self.btn_register.setEnabled(True),
                                     QMessageBox.critical(self, "Error", e)))
        def worker():
            try:
                from atlas.registration import MNIRegistration
                r = MNIRegistration().register(t1_path, mask_path, out_dir, reg_type)
                sig.finished.emit(r)
            except Exception as e:
                sig.error.emit(str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _on_registration_done(self, reg_result):
        self.prog_atlas.hide(); self.btn_register.setEnabled(True)
        self._mask_mni_path = reg_result["mask_mni"]
        self.txt_atlas_results.setText("✔ Registration complete. Compute overlap.")
        if self.result is not None:
            self.btn_analyze_atlas.setEnabled(True)

    def _run_atlas_analysis(self):
        if self._mask_mni_path is None or self.result is None:
            QMessageBox.warning(self, "Prerequisites", "Segment and register first."); return
        choice = self.cb_atlas_mask.currentText()
        if "Hyper" in choice:
            mask_arr, mname = self.result.hyper_mask.astype(np.uint8), "hyper"
        elif "Hypo" in choice:
            mask_arr, mname = self.result.hypo_mask.astype(np.uint8), "hypo"
        else:
            mask_arr, mname = self.data.mask.astype(np.uint8), "whole"
        mask_path = OUTPUT_DIR / self.data.patient_id / f"mask_{mname}.nii.gz"
        save_nifti_like(self.data.reference_img, mask_arr, mask_path)
        self.txt_atlas_results.setText(f"⏳ Atlas analysis on {mname}...")
        sig = _Signals()
        sig.finished.connect(lambda r: self._on_atlas_done(r, mname))
        sig.error.connect(lambda e: QMessageBox.critical(self, "Atlas error", e))
        t1_path = self._t1_path or (OUTPUT_DIR / self.data.patient_id / "T1_temp.nii.gz")
        out_dir = OUTPUT_DIR / self.data.patient_id / "mni_registration"
        reg_type = "Affine" if "Affine" in self.cb_reg_type.currentText() else "SyN"
        def worker():
            try:
                from atlas.registration import MNIRegistration
                from atlas.wm_report import WMAtlasAnalyzer
                r = MNIRegistration().register(t1_path, mask_path, out_dir, reg_type)
                rep = WMAtlasAnalyzer().analyze(r["mask_mni"], self.data.patient_id,
                                                self.result.model_name, self.result.n_clusters)
                sig.finished.emit(rep)
            except Exception as e:
                sig.error.emit(str(e))
        threading.Thread(target=worker, daemon=True).start()

    def _on_atlas_done(self, report, mname=""):
        self._atlas_report = report
        self.txt_atlas_results.setText(f"— {mname} —\n\n" + report.summary_text())
        self.btn_save_atlas_csv.setEnabled(True)

    def _save_atlas_csv(self):
        if self._atlas_report is None:
            return
        out = OUTPUT_DIR / self.data.patient_id / "atlas_wm_report.csv"
        from atlas.wm_report import WMAtlasAnalyzer
        WMAtlasAnalyzer().save_report_csv(self._atlas_report, out)
        QMessageBox.information(self, "Saved", f"CSV:\n{out}")

    # ── Azioni ───────────────────────────────────────────────────────────────

    def _validate_result(self):
        if self.run_id:
            self.db.mark_validated(self.run_id)
            QMessageBox.information(self, "Validated", f"Run #{self.run_id} validated.")

    def _save_result(self):
        if self.result is None or self.data is None:
            return
        out_dir = OUTPUT_DIR / self.data.patient_id
        out_dir.mkdir(parents=True, exist_ok=True)
        ref = self.data.reference_img
        save_nifti_like(ref, self.result.label_volume,
                        out_dir / f"clusters_{self.result.model_name}.nii.gz")
        save_nifti_like(ref, self.result.hypo_mask.astype(np.uint8), out_dir/"hypo.nii.gz")
        save_nifti_like(ref, self.result.hyper_mask.astype(np.uint8), out_dir/"hyper.nii.gz")
        QMessageBox.information(self, "Saved", f"Output in:\n{out_dir}")

    def _refresh_history(self):
        patient = self.cb_patient.currentText()
        if not patient:
            return
        history = self.db.get_patient_history(patient)
        if not history:
            self.txt_history.setText("No run."); return
        lines = [f"History — {patient}\n"]
        for r in history:
            v = "✔" if r["is_validated"] else "○"
            lines.append(f"{v} [{r['created_at'][:16]}] {r['model_name']} "
                         f"k={r['n_clusters']} #{r['id']}")
        self.txt_history.setText("\n".join(lines))

    # ── Agenti ───────────────────────────────────────────────────────────────

    def _agent_run_async(self, fn, *args):
        # I widget Qt vanno aggiornati solo dal main thread: il worker calcola
        # in background e consegna il risultato via Signal (come la segmentazione).
        self.txt_agent.setText("⏳ Processing...")
        sig = _Signals()
        sig.finished.connect(self.txt_agent.setText)
        sig.error.connect(lambda e: self.txt_agent.setText(f"[Error: {e}]"))

        def worker():
            try:
                sig.finished.emit(fn(*args))
            except Exception as e:
                import traceback; traceback.print_exc()
                sig.error.emit(str(e))

        self._agent_thread = threading.Thread(target=worker, daemon=True)
        self._agent_thread.start()

    def _agent_suggest_methods(self):
        current = [m["name"] for m in list_models()]
        modality = self.data.modality.value if self.data else "PET"
        self._agent_run_async(self._agents["research"].suggest_methods,
                              f"{modality} co-registered", 10, current)

    def _agent_interpret(self):
        if self.result is None:
            self.txt_agent.setText("Run a segmentation first."); return
        self._agent_run_async(self._agents["model"].interpret_metrics,
                              self.data.patient_id, self.result.model_name,
                              {k: v for k, v in self.result.metrics.items()
                               if not isinstance(v, (list, dict))})

    def _agent_report(self):
        if self.result is None or self.data is None:
            self.txt_agent.setText("Run a segmentation first."); return
        wm = self._atlas_report.to_dict_list()[:10] if self._atlas_report else None
        self._agent_run_async(self._agents["model"].generate_clinical_report,
                              self.data.patient_id, self.result.model_name,
                              self.result.n_clusters,
                              {k: v for k, v in self.result.metrics.items()
                               if not isinstance(v, (list, dict))}, wm)
