"""
app.py — Entry point di GlioCore.

Avvia napari con il pannello GlioCore agganciato come dock widget.

Utilizzo:
    python app.py
"""
import logging
import sys
from pathlib import Path

# Aggiungi la root del progetto al PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

import napari
from config.settings import APP_TITLE, DATA_DIR, OUTPUT_DIR, ATLAS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gliocore")


def main():
    # Crea cartelle se non esistono
    for d in [DATA_DIR, OUTPUT_DIR, ATLAS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting {APP_TITLE}")
    log.info(f"Patient data: {DATA_DIR}")
    log.info(f"Output:        {OUTPUT_DIR}")

    # Crea il viewer napari
    viewer = napari.Viewer(title=APP_TITLE)

    # Importa e aggancia il widget principale
    from ui.main_widget import GlioCoreWidget
    widget = GlioCoreWidget(viewer)
    viewer.window.add_dock_widget(
        widget,
        name="GlioCore",
        area="right",
    )

    log.info("UI ready. Enjoy!")
    napari.run()


if __name__ == "__main__":
    main()
