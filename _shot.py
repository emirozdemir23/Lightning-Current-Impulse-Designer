import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from PyQt6.QtWidgets import QApplication
from src.gui import MainWindow

flat = json.load(open("config.json", encoding="utf-8"))["impulse_generator"]
app = QApplication.instance() or QApplication(sys.argv)
win = MainWindow(initial_config=flat)
win.show()
for _ in range(10): app.processEvents()
win.grab().save("_rc.png", "PNG")
win.mode_combo.setCurrentIndex(1)
for _ in range(10): app.processEvents()
win.grab().save("_rlc.png", "PNG")
win.mode_combo.setCurrentIndex(2)
for _ in range(10): app.processEvents()
win.grab().save("_ic.png", "PNG")
win.close()
