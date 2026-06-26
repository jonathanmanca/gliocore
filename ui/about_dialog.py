"""
ui/about_dialog.py — Finestra About di GlioLab (versione curata).
"""
from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QDesktopServices, QFont
from qtpy.QtCore import QUrl
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QWidget,
)

APP_VERSION = "1.0.0"


class AboutDialog(QDialog):
    """Finestra informativa su GlioLab, con stile coerente al brand."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About GlioCore")
        self.setMinimumWidth(440)
        self.setMaximumWidth(480)
        self._build()

    def _build(self):
        self.setStyleSheet("""
            QDialog { background: #FFFFFF; }
            QLabel { color: #1c3550; background: transparent; }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Header con sfondo blu ────────────────────────────────────────────
        header = QWidget()
        header.setStyleSheet("background: #1A3A5C;")
        header.setAutoFillBackground(True)
        hl = QVBoxLayout(header)
        hl.setContentsMargins(30, 24, 30, 24)
        hl.setSpacing(4)

        title = QLabel("GlioCore")
        f = QFont("Arial", 30); f.setBold(True)
        title.setFont(f)
        title.setStyleSheet("color: #FFFFFF; background: transparent;")
        title.setAlignment(Qt.AlignCenter)
        hl.addWidget(title)

        subtitle = QLabel("Metabolic Glioma Segmentation from PET/MRI")
        subtitle.setStyleSheet("color: #AED6F1; font-size: 12px; background: transparent;")
        subtitle.setAlignment(Qt.AlignCenter)
        hl.addWidget(subtitle)

        ver = QLabel(f"Version {APP_VERSION}")
        ver.setStyleSheet("color: #7FB3D5; font-size: 11px; background: transparent;")
        ver.setAlignment(Qt.AlignCenter)
        hl.addWidget(ver)

        root.addWidget(header)

        # ── Corpo ────────────────────────────────────────────────────────────
        body = QWidget()
        body.setObjectName("bodyPanel")
        body.setStyleSheet("#bodyPanel { background: #FFFFFF; }")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(30, 22, 30, 22)
        bl.setSpacing(14)

        desc = QLabel(
            "Open-source tool for semi-automatic glioma segmentation "
            "from co-registered PET and MRI images. It combines six "
            "clustering models, a white-matter atlas, active learning, "
            "and three operating modalities (PET, MRI, PET+MRI)."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("font-size: 12px; color: #2C3E50; background: transparent;")
        bl.addWidget(desc)

        bl.addWidget(self._separator())

        # Author
        author_row = QVBoxLayout()
        author_row.setSpacing(2)
        au = QLabel("Jonathan Manca")
        au.setStyleSheet("font-size: 14px; font-weight: bold; color: #1A3A5C;")
        author_row.addWidget(au)
        role = QLabel("Independent researcher · Neuro-oncology imaging")
        role.setStyleSheet("font-size: 11px; color: #7F8C8D;")
        author_row.addWidget(role)
        bl.addLayout(author_row)

        # LinkedIn button
        btn_linkedin = QPushButton("  Open LinkedIn profile")
        btn_linkedin.setCursor(Qt.PointingHandCursor)
        btn_linkedin.setStyleSheet("""
            QPushButton {
                background: #0077B5; color: white; border: none;
                border-radius: 5px; padding: 8px 14px; font-size: 12px;
                font-weight: bold; text-align: center;
            }
            QPushButton:hover { background: #005f91; }
        """)
        btn_linkedin.clicked.connect(
            lambda: QDesktopServices.openUrl(
                QUrl("https://www.linkedin.com/in/jonathan-manca"))
        )
        bl.addWidget(btn_linkedin)

        bl.addWidget(self._separator())

        # License + disclaimer
        lic = QLabel("MIT License · © 2026 Jonathan Manca")
        lic.setStyleSheet("font-size: 11px; color: #7F8C8D;")
        lic.setAlignment(Qt.AlignCenter)
        bl.addWidget(lic)

        disclaimer = QLabel(
            "⚠  Research tool only. Not a certified medical device; "
            "it must not be used for clinical decisions without "
            "qualified medical supervision."
        )
        disclaimer.setWordWrap(True)
        disclaimer.setStyleSheet(
            "font-size: 10px; color: #A93226; background: #FDEDEC; "
            "border-radius: 4px; padding: 8px;"
        )
        bl.addWidget(disclaimer)

        root.addWidget(body)

        # ── Footer con pulsante chiudi ───────────────────────────────────────
        footer = QHBoxLayout()
        footer.setContentsMargins(30, 0, 30, 18)
        footer.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("""
            QPushButton {
                background: #ECF0F1; color: #2C3E50; border: none;
                border-radius: 5px; padding: 7px 20px; font-size: 12px;
            }
            QPushButton:hover { background: #D5DBDB; }
        """)
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)
        root.addLayout(footer)

    def _separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #E5E8E8; background: #E5E8E8; max-height: 1px;")
        return line
