"""PyQt6 GUI for the High Voltage Transient Analyzer.

Layout::

    +----------------------+--------------------------------+
    |  LEFT PANEL          |  RIGHT PANEL                   |
    |  (fixed 320 px)      |                                |
    |                      |                                |
    |  Circuit Config      |  +--------------------------+  |
    |  [QComboBox]         |  |                          |  |
    |  V0 label            |  |    Matplotlib canvas     |  |
    |  [QDoubleSpinBox]    |  |                          |  |
    |   ...                |  +--------------------------+  |
    |  L  (RLC only)       |  +--------------------------+  |
    |  [QDoubleSpinBox]    |  | Vmax  |  T1   |   T2     |  |
    |  (stretch)           |  | [   PASS / FAIL status  ] |  |
    |  [Run Simulation]    |  +--------------------------+  |
    +----------------------+--------------------------------+

The window owns one Matplotlib Figure; ``run_simulation`` calls the
shared ``plotter.draw_impulse_response`` helper to redraw the embedded
canvas in place. Simulation / evaluation / visualization logic is never
duplicated here.
"""

import sys
import traceback

import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout,
    QLabel, QDoubleSpinBox, QComboBox, QPushButton,
    QFrame, QMessageBox, QTabWidget,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Rectangle

from src.solver import solve_impulse_circuit, find_optimal_resistors
from src.standards import evaluate_lightning_impulse, evaluate_current_impulse


_LEFT_PANEL_WIDTH = 320

# (config_key, label_text, suffix, min, max, step, decimals).
# config_keys match the JSON schema in config.json so the dict returned
# by _current_config() drops straight into solve_impulse_circuit.
_INPUTS = [
    ("V0_kV",  "V0  —  Charging Voltage",    " kV",   1.0,    5000.0,    10.0, 2),
    ("C1_nF",  "C1  —  Storage Capacitor",   " nF",   0.01,  10000.0,     1.0, 2),
    ("C2_nF",  "C2  —  Load Capacitor",      " nF",   0.01,   1000.0,     0.1, 2),
    ("R1_ohm", "R1  —  Front Resistor",      " Ω",    0.1,  100000.0,    50.0, 1),
    ("R2_ohm", "R2  —  Tail Resistor",       " Ω",    0.1,  100000.0,   100.0, 1),
    ("L_uH",   "L  —  Stray Inductance",     " μH",   0.01,  10000.0,     1.0, 2),
]

# (config_value, display_text). Order defines combo-box index.
_MODE_OPTIONS = [
    ("RC",  "Standard RC"),
    ("RLC", "RLC (With Inductance)"),
    ("Impulse Current", "Impulse Current Generator (8/20 µs)"),
]


class MainWindow(QMainWindow):
    """Top-level analyzer window."""

    def __init__(self, initial_config=None):
        super().__init__()
        self.setWindowTitle("HV Transient Analyzer")
        self.resize(1280, 760)

        self.spinboxes = {}
        # Latest evaluation results, cached so the status bar can be
        # refreshed on tab switches without re-running the simulation.
        self._last_ev_v = None
        self._last_ev_c = None
        self._build_ui()

        if initial_config is not None:
            self._apply_config(initial_config)
        else:
            self._sync_mode_dependent_widgets()
        # No explicit run_simulation() here — _sync_mode_dependent_widgets
        # (called from both branches above) already refreshes the
        # canvases so the startup state is rendered exactly once.

    # ------------------------------------------------------------------ layout

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(14)
        layout.addWidget(self._build_left_panel(), 0)
        layout.addWidget(self._build_right_panel(), 1)
        self.setCentralWidget(central)

    def _build_left_panel(self):
        panel = QFrame()
        panel.setFixedWidth(_LEFT_PANEL_WIDTH)
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(4)

        header = QLabel("Impulse Generator")
        header.setStyleSheet(
            "font-size: 14pt; font-weight: 700; padding-bottom: 8px;"
        )
        layout.addWidget(header)

        self.schematic_fig = Figure(figsize=(3, 2), facecolor="white")
        self.schematic_canvas = FigureCanvas(self.schematic_fig)
        self.schematic_canvas.setFixedHeight(170)
        self.schematic_canvas.setStyleSheet(
            "background-color: white;"
            " border: 1px solid #cbd5e1;"
            " border-radius: 6px;"
        )
        layout.addWidget(self.schematic_canvas)

        mode_label = QLabel("Circuit Configuration")
        mode_label.setStyleSheet("font-weight: 600; margin-top: 4px;")
        self.mode_combo = QComboBox()
        for _, text in _MODE_OPTIONS:
            self.mode_combo.addItem(text)
        self.mode_combo.setMinimumHeight(30)
        self.mode_combo.currentIndexChanged.connect(
            self._sync_mode_dependent_widgets)
        layout.addWidget(mode_label)
        layout.addWidget(self.mode_combo)

        for key, text, suffix, mn, mx, step, decimals in _INPUTS:
            label = QLabel(text)
            label.setStyleSheet("font-weight: 600; margin-top: 8px;")
            spin = QDoubleSpinBox()
            spin.setRange(mn, mx)
            spin.setSingleStep(step)
            spin.setDecimals(decimals)
            spin.setSuffix(suffix)
            spin.setMinimumHeight(30)
            layout.addWidget(label)
            layout.addWidget(spin)
            self.spinboxes[key] = spin

        layout.addStretch(1)

        self.tune_btn = QPushButton("Auto-Tune to Standard (1.2/50 µs)")
        self.tune_btn.setMinimumHeight(36)
        self.tune_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #f0f4f8;"
            "  color: #2c6fb0;"
            "  font-weight: 600; font-size: 10.5pt;"
            "  border: 1.5px solid #2c6fb0;"
            "  border-radius: 6px;"
            "  padding: 6px;"
            "}"
            "QPushButton:hover  { background-color: #e1eaf3; }"
            "QPushButton:pressed{ background-color: #c9d8e6; }"
            "QPushButton:disabled {"
            "  color: #a0a0a0;"
            "  border-color: #c8c8c8;"
            "  background-color: #f5f5f5;"
            "}"
        )
        self.tune_btn.clicked.connect(self.trigger_auto_tune)
        layout.addWidget(self.tune_btn)

        self.run_btn = QPushButton("Run Simulation")
        self.run_btn.setMinimumHeight(44)
        self.run_btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #2c6fb0;"
            "  color: white;"
            "  font-weight: 700; font-size: 12pt;"
            "  border: none; border-radius: 6px;"
            "}"
            "QPushButton:hover  { background-color: #1f5c97; }"
            "QPushButton:pressed{ background-color: #154a7a; }"
        )
        self.run_btn.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_btn)
        return panel

    def _build_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.tabs = QTabWidget()
        self.tabs.addTab(
            self._build_voltage_tab(),
            "Darbe Gerilimi (Voltage Analysis)")
        self.tabs.addTab(
            self._build_current_tab(),
            "Darbe Akımı (Current Analysis)")
        # Refresh the status bar when the active tab changes so it always
        # reflects the standard for the tab currently in view.
        self.tabs.currentChanged.connect(self._update_status_bar)
        layout.addWidget(self.tabs, 1)

        layout.addWidget(self._build_results_panel(), 0)
        return panel

    def _build_voltage_tab(self):
        w = QWidget()
        L = QVBoxLayout(w)
        L.setContentsMargins(8, 8, 8, 8)
        L.setSpacing(8)

        self.fig_v = Figure(figsize=(7, 4))
        self.canvas_v = FigureCanvas(self.fig_v)
        self.ax_volt = self.fig_v.add_subplot(111)
        L.addWidget(self.canvas_v, 1)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.lbl_vmax = self._metric_label("Vmax", "—")
        self.lbl_t1_v = self._metric_label("T1 (Voltage)", "—")
        self.lbl_t2_v = self._metric_label("T2 (Voltage)", "—")
        for c in (self.lbl_vmax, self.lbl_t1_v, self.lbl_t2_v):
            row.addWidget(c, 1)
        L.addLayout(row)
        return w

    def _build_current_tab(self):
        w = QWidget()
        L = QVBoxLayout(w)
        L.setContentsMargins(8, 8, 8, 8)
        L.setSpacing(8)

        self.fig_c = Figure(figsize=(7, 4))
        self.canvas_c = FigureCanvas(self.fig_c)
        self.ax_curr = self.fig_c.add_subplot(111)
        L.addWidget(self.canvas_c, 1)

        row = QHBoxLayout()
        row.setSpacing(12)
        self.lbl_imax = self._metric_label("Imax", "—")
        self.lbl_t1_c = self._metric_label("T1 (Current)", "—")
        self.lbl_t2_c = self._metric_label("T2 (Current)", "—")
        for c in (self.lbl_imax, self.lbl_t1_c, self.lbl_t2_c):
            row.addWidget(c, 1)
        L.addLayout(row)
        return w

    def _build_results_panel(self):
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)
        panel.setStyleSheet("QFrame { background: #fafbfc; }")
        outer = QVBoxLayout(panel)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        self.lbl_status = QLabel("AWAITING SIMULATION")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setMinimumHeight(48)
        self._set_status_style("neutral")
        outer.addWidget(self.lbl_status)

        return panel

    def _metric_label(self, name, value):
        lbl = QLabel()
        lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(
            "QLabel {"
            " background-color: #f1f5f9;"
            " color: #0f172a;"
            " border: 1px solid #cbd5e1;"
            " border-radius: 6px;"
            " padding: 8px;"
            "}"
        )
        self._set_metric_text(lbl, name, value)
        return lbl

    def _set_metric_text(self, label, name, value):
        label.setText(
            f"<span style='color:#475569; font-size:10pt; "
            f"font-weight:600;'>{name}</span><br>"
            f"<b style='color:#0f172a; font-size:15pt;'>{value}</b>"
        )

    def _set_status_style(self, kind):
        colors = {"pass": "#2e7d32", "fail": "#c62828", "neutral": "#777777"}
        bg = colors.get(kind, "#777777")
        self.lbl_status.setStyleSheet(
            f"QLabel {{"
            f"  background-color: {bg};"
            f"  color: white; font-weight: 700; font-size: 13pt;"
            f"  letter-spacing: 1px; border-radius: 6px;"
            f"}}"
        )

    # ----------------------------------------------------- circuit-mode toggle

    def _circuit_mode(self):
        """Return the canonical config value ('RC' or 'RLC') for the
        currently selected combo-box item."""
        return _MODE_OPTIONS[self.mode_combo.currentIndex()][0]

    def _set_circuit_mode(self, mode):
        for idx, (m, _) in enumerate(_MODE_OPTIONS):
            if m == mode:
                self.mode_combo.setCurrentIndex(idx)
                return
        # Unknown mode → leave the combo where it is.

    def _tune_label_for(self, mode):
        """Mode-appropriate label for the Auto-Tune button."""
        if mode == "Impulse Current":
            return "Auto-Tune to Standard (8/20 µs)"
        return "Auto-Tune to Standard (1.2/50 µs)"

    def _sync_mode_dependent_widgets(self, *_args):
        """Update widgets whose enabled state depends on the circuit mode,
        refresh the schematic + Auto-Tune label, and re-run the
        simulation so the plot never displays stale-mode curves.

        * L spinbox enabled in RLC and Impulse Current modes.
        * C2 / R2 spinboxes disabled in Impulse Current mode.
        * Auto-Tune is enabled for every mode; its label tracks the
          mode-specific target (1.2/50 µs or 8/20 µs).
        """
        mode = self._circuit_mode()
        is_current = mode == "Impulse Current"
        self.spinboxes["L_uH"].setEnabled(mode in ("RLC", "Impulse Current"))
        self.spinboxes["C2_nF"].setEnabled(not is_current)
        self.spinboxes["R2_ohm"].setEnabled(not is_current)
        self.tune_btn.setEnabled(True)
        self.tune_btn.setText(self._tune_label_for(mode))
        self.draw_circuit_schematic(mode)
        # Refresh the canvases so a mode switch never leaves stale curves.
        self.run_simulation()

    # --------------------------------------------------------------- state I/O

    def _apply_config(self, config):
        for key, *_ in _INPUTS:
            if key in config:
                self.spinboxes[key].setValue(float(config[key]))
        self._set_circuit_mode(config.get("circuit_mode", "RC"))
        # setCurrentIndex doesn't emit when the index is unchanged, so
        # re-sync the enabled state explicitly.
        self._sync_mode_dependent_widgets()

    def _current_config(self):
        params = {key: self.spinboxes[key].value() for key, *_ in _INPUTS}
        params["circuit_mode"] = self._circuit_mode()
        return params

    # ---------------------------------------------------------------- handler

    def _reconstruct_V_C1(self, t, current):
        """V_C1(t) recovered from V0, C1, and the loop current via
        charge conservation::

            V_C1(t) = V0 - (1/C1) * ∫₀ᵗ i(τ) dτ

        Unit factor: kA·μs / nF = 1e3 kV.
        """
        V0 = self.spinboxes["V0_kV"].value()
        C1 = self.spinboxes["C1_nF"].value()
        if len(t) < 2:
            return np.full_like(t, V0)
        dx = np.diff(t)
        y_avg = 0.5 * (current[:-1] + current[1:])
        integ = np.concatenate(([0.0], np.cumsum(y_avg * dx)))
        return V0 - integ * 1e3 / C1

    def _render_plots(self, t, voltage, current, ev_v, ev_c):
        """Tab 1 (Voltage): V_C1 dashed + V_out solid on self.ax_volt.
        Tab 2 (Current): i_L on self.ax_curr.
        Each axis carries its own reference dotted lines and metrics
        box for its respective standard.
        """
        self.ax_volt.clear()
        self.ax_curr.clear()

        V_C1 = self._reconstruct_V_C1(t, current)

        self.ax_volt.plot(
            t, V_C1, color="#4a6fa5", linewidth=1.4,
            linestyle="--", alpha=0.85,
            label=r"$V_{C1}(t)$ — storage cap")
        self.ax_volt.plot(
            t, voltage, color="#c44536", linewidth=2.0,
            label=r"$V_{out}(t)$ — output")

        Vmax = ev_v["Vmax_kV"]
        for frac in (1.0, 0.9, 0.5, 0.3):
            self.ax_volt.axhline(frac * Vmax, color="gray", linestyle=":",
                                 linewidth=0.8, alpha=0.55)

        self.ax_volt.text(
            0.97, 0.95,
            f"$V_{{max}}$ = {Vmax:.2f} kV\n"
            f"$T_1$ = {ev_v['T1_us']:.3f} μs\n"
            f"$T_2$ = {ev_v['T2_us']:.2f} μs\n"
            f"IEC 1.2/50: "
            f"{'PASS' if ev_v['is_standard_lightning'] else 'FAIL'}",
            transform=self.ax_volt.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", alpha=0.92),
        )

        self.ax_volt.set_xlabel("Time (μs)")
        self.ax_volt.set_ylabel("Voltage (kV)")
        self.ax_volt.set_title(
            "Darbe Gerilim Yanıtı (Impulse Voltage Response)")
        self.ax_volt.grid(True, alpha=0.3)
        self.ax_volt.legend(loc="upper right", fontsize=9)

        self.ax_curr.plot(
            t, current, color="#1f8a3d", linewidth=1.8,
            label=r"$i_L(t)$ — loop current")

        Imax = ev_c["Imax_kA"]
        for frac in (1.0, 0.9, 0.5, 0.1):
            self.ax_curr.axhline(frac * Imax, color="gray", linestyle=":",
                                 linewidth=0.8, alpha=0.55)

        self.ax_curr.text(
            0.97, 0.95,
            f"$I_{{max}}$ = {Imax:.2f} kA\n"
            f"$T_1$ = {ev_c['T1_us']:.3f} μs\n"
            f"$T_2$ = {ev_c['T2_us']:.2f} μs\n"
            f"IEC 8/20: "
            f"{'PASS' if ev_c['is_standard_8_20'] else 'FAIL'}",
            transform=self.ax_curr.transAxes,
            ha="right", va="top", fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.4",
                      facecolor="white", edgecolor="gray", alpha=0.92),
        )

        self.ax_curr.set_xlabel("Time (μs)")
        self.ax_curr.set_ylabel("Current (kA)")
        self.ax_curr.set_title(
            "Darbe Akım Yanıtı (Impulse Current Response)")
        self.ax_curr.grid(True, alpha=0.3)
        self.ax_curr.legend(loc="upper right", fontsize=9)

    def run_simulation(self):
        """Read inputs, drive solver + both standards, refresh both
        tab canvases, their dedicated result cards, and the global
        compliance status bar."""
        config = self._current_config()
        mode = config["circuit_mode"]

        # Input validation: the storage capacitor C1 must be at least
        # 10x the load capacitor C2 so it behaves as a near-ideal source
        # while charging C2. Skip in Impulse Current mode, where C2 is
        # disabled and not part of the circuit.
        if mode != "Impulse Current" and config["C1_nF"] < 10 * config["C2_nF"]:
            QMessageBox.critical(
                self, "Hata",
                "Hata: Depolama kapasitesi (C1), yük kapasitesinin (C2) "
                "en az 10 katı olmalıdır!",
            )
            return

        try:
            t, voltage, current = solve_impulse_circuit(config)
            ev_v = evaluate_lightning_impulse(t, voltage)
            ev_c = evaluate_current_impulse(t, current)
        except Exception as exc:
            QMessageBox.critical(
                self, "Simulation Error",
                f"Could not complete the simulation:\n\n{exc}",
            )
            return

        self._render_plots(t, voltage, current, ev_v, ev_c)
        self.fig_v.tight_layout()
        self.fig_c.tight_layout()
        self.canvas_v.draw_idle()
        self.canvas_c.draw_idle()

        self._set_metric_text(
            self.lbl_vmax, "Vmax", f"{ev_v['Vmax_kV']:.2f} kV")
        self._set_metric_text(
            self.lbl_t1_v, "T1 (Voltage)", f"{ev_v['T1_us']:.3f} μs")
        self._set_metric_text(
            self.lbl_t2_v, "T2 (Voltage)", f"{ev_v['T2_us']:.2f} μs")
        self._set_metric_text(
            self.lbl_imax, "Imax", f"{ev_c['Imax_kA']:.2f} kA")
        self._set_metric_text(
            self.lbl_t1_c, "T1 (Current)", f"{ev_c['T1_us']:.3f} μs")
        self._set_metric_text(
            self.lbl_t2_c, "T2 (Current)", f"{ev_c['T2_us']:.2f} μs")

        # Cache the results and let the status bar reflect the active tab.
        self._last_ev_v = ev_v
        self._last_ev_c = ev_c
        self._update_status_bar()

    def _update_status_bar(self, *_args):
        """Refresh the compliance status bar so it reflects the standard
        for the tab currently in view: the Voltage Analysis tab reports
        the IEC 1.2/50 lightning result, while the Current Analysis tab
        reports the IEC 8/20 current result.

        Connected to ``self.tabs.currentChanged`` and also called at the
        end of ``run_simulation``; a no-op until the first simulation has
        populated the cached evaluations.
        """
        if self._last_ev_v is None or self._last_ev_c is None:
            return

        # Tab index 1 is "Darbe Akımı (Current Analysis)".
        if self.tabs.currentIndex() == 1:
            is_pass = self._last_ev_c["is_standard_8_20"]
            pass_text = "PASS  —  IEC 60060-1 STANDARD 8/20"
            fail_text = "FAIL  —  NON-STANDARD CURRENT WAVEFORM"
        else:
            is_pass = self._last_ev_v["is_standard_lightning"]
            pass_text = "PASS  —  IEC 60060-1 STANDARD 1.2/50"
            fail_text = "FAIL  —  NON-STANDARD WAVEFORM"

        if is_pass:
            self.lbl_status.setText(pass_text)
            self._set_status_style("pass")
        else:
            self.lbl_status.setText(fail_text)
            self._set_status_style("fail")

    def draw_circuit_schematic(self, mode=None):
        """Render a small topology preview into the left-panel canvas.

        ``"RC"`` (Marx-equivalent) shows the multi-stage parameter
        formulas next to V0/C1/R1/R2. ``"RLC"`` and ``"Impulse Current"``
        show the same loop with the stray inductor inserted; the
        impulse-current variant omits the C2/R2 load branch.
        """
        if mode is None:
            mode = self._circuit_mode()
        fig = self.schematic_fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

        LC = "#1e293b"
        LW = 1.3
        TXT = dict(fontsize=7, color="#0f172a")

        def hwire(x1, x2, y):
            ax.plot([x1, x2], [y, y], color=LC, linewidth=LW)

        def vwire(x, y1, y2):
            ax.plot([x, x], [y1, y2], color=LC, linewidth=LW)

        def cap(x, y_top, y_bot):
            ym = (y_top + y_bot) / 2
            vwire(x, y_top, ym + 0.18)
            hwire(x - 0.35, x + 0.35, ym + 0.18)
            hwire(x - 0.35, x + 0.35, ym - 0.18)
            vwire(x, ym - 0.18, y_bot)

        def box(x, y, w, h, label=""):
            ax.add_patch(Rectangle(
                (x, y), w, h, fill=True,
                facecolor="white", edgecolor=LC, linewidth=LW))
            if label:
                ax.text(x + w / 2, y + h / 2, label,
                        ha="center", va="center",
                        fontsize=7, color=LC, fontweight="bold")

        y_top, y_bot = 4.0, 1.0

        if mode == "RC":
            hwire(2.0, 3.4, y_top)
            hwire(6.6, 8.5, y_top)
            hwire(2.0, 8.5, y_bot)
            cap(2.0, y_top, y_bot)
            box(3.4, y_top - 0.25, 3.2, 0.5)
            cap(7.0, y_top, y_bot)
            box(8.25, 1.8, 0.5, 1.5)
            vwire(8.5, y_top, 3.3)
            vwire(8.5, 1.8, y_bot)
            ax.text(2.0, 5.4, "V0 = n * V_stage",
                    ha="center", va="bottom", **TXT)
            ax.text(2.0, 0.5, "C1 = C_stage / n",
                    ha="center", va="top", **TXT)
            ax.text(5.0, 4.55, "R1 = n * R_front",
                    ha="center", va="bottom", **TXT)
            ax.text(7.0, 0.5, "C2",
                    ha="center", va="top", **TXT)
            ax.text(8.5, 0.5, "R2 = n * R_tail",
                    ha="center", va="top", **TXT)

        elif mode == "RLC":
            hwire(2.0, 3.0, y_top)
            hwire(4.2, 5.4, y_top)
            hwire(6.6, 7.5, y_top)
            hwire(2.0, 9.0, y_bot)
            cap(2.0, y_top, y_bot)
            box(3.0, y_top - 0.25, 1.2, 0.5, "R1")
            box(5.4, y_top - 0.25, 1.2, 0.5, "L")
            cap(7.5, y_top, y_bot)
            hwire(7.5, 9.0, y_top)
            box(8.75, 1.8, 0.5, 1.5)
            vwire(9.0, y_top, 3.3)
            vwire(9.0, 1.8, y_bot)
            ax.text(2.0, 5.4, "V0", ha="center", va="bottom", **TXT)
            ax.text(2.0, 0.5, "C1", ha="center", va="top", **TXT)
            ax.text(7.5, 5.4, "C2", ha="center", va="bottom", **TXT)
            ax.text(9.0, 0.5, "R2", ha="center", va="top", **TXT)

        elif mode == "Impulse Current":
            hwire(2.0, 3.5, y_top)
            hwire(4.7, 5.9, y_top)
            hwire(7.1, 8.5, y_top)
            hwire(2.0, 8.5, y_bot)
            cap(2.0, y_top, y_bot)
            box(3.5, y_top - 0.25, 1.2, 0.5, "R1")
            box(5.9, y_top - 0.25, 1.2, 0.5, "L")
            vwire(8.5, y_top, y_bot)
            ax.text(2.0, 5.4, "V0", ha="center", va="bottom", **TXT)
            ax.text(2.0, 0.5, "C1", ha="center", va="top", **TXT)
            ax.text(8.5, 5.4, "Load (i_L)",
                    ha="center", va="bottom", **TXT)

        fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
        self.schematic_canvas.draw_idle()

    def trigger_auto_tune(self):
        """Optimize R1, R2 for the 1.2/50 lightning impulse, push the
        result into the UI, then re-run the simulation so the canvas
        and compliance panel refresh in a single click.

        The whole body is wrapped in try/except with traceback printing
        so any failure (in the optimizer, in setValue, in run_simulation,
        anywhere) lands in the terminal instead of being swallowed by
        the Qt event loop.
        """
        mode = self._circuit_mode()
        is_current = mode == "Impulse Current"
        print(f"[Auto-Tune] Starting optimization for mode={mode!r}...",
              flush=True)

        # Physical-impossibility guard (voltage modes only): the
        # 1.2/50 front network requires C1 >> C2 so the storage cap
        # behaves as a near-ideal source while charging C2 through R1.
        if not is_current:
            c1 = self.spinboxes["C1_nF"].value()
            c2 = self.spinboxes["C2_nF"].value()
            if c1 <= c2:
                QMessageBox.warning(
                    self, "Physical Impossibility",
                    "Fiziksel Sınır Aşımı:\n\n"
                    "Şarj kapasitörü (C1) yük kapasitöründen (C2) büyük "
                    "olmalıdır! Bu oranlarla standart 1.2/50 µs dalgası "
                    "üretmek fiziksel olarak imkansızdır."
                )
                return

        self.tune_btn.setText("Tuning...")
        self.tune_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        if is_current:
            a_initial = self.spinboxes["R1_ohm"].value()
            b_initial = self.spinboxes["L_uH"].value()
        else:
            a_initial = self.spinboxes["R1_ohm"].value()
            b_initial = self.spinboxes["R2_ohm"].value()

        try:
            config = self._current_config()
            print(
                f"[Auto-Tune] Input  V0={config['V0_kV']:.3f} kV  "
                f"C1={config['C1_nF']:.4f} nF  C2={config['C2_nF']:.4f} nF  "
                f"R1={config['R1_ohm']:.3f} ohm  R2={config['R2_ohm']:.3f} ohm  "
                f"L={config['L_uH']:.4f} uH  mode={config['circuit_mode']}",
                flush=True,
            )

            if is_current:
                opt = find_optimal_resistors(
                    config, target_T1=8.0, target_T2=20.0)
                R1_opt = float(opt["R1_ohm"])
                L_opt = float(opt["L_uH"])
                print(
                    f"[Auto-Tune] Result R1={R1_opt:.3f} ohm  "
                    f"L={L_opt:.4f} uH",
                    flush=True,
                )
                if (abs(R1_opt - a_initial) < 1e-3
                        and abs(L_opt - b_initial) < 1e-3):
                    QMessageBox.warning(
                        self, "Optimization Failed",
                        "Optimizasyon Başarısız:\n\n"
                        "Seçilen parametreler ile standart 8/20 µs akım "
                        "dalgası elde edilebilecek uygun bir R1/L "
                        "kombinasyonu bulunamadı."
                    )
                    return
                self.spinboxes["R1_ohm"].setValue(R1_opt)
                self.spinboxes["L_uH"].setValue(L_opt)
            else:
                opt = find_optimal_resistors(
                    config, target_T1=1.2, target_T2=50.0)
                R1_opt = float(opt["R1_ohm"])
                R2_opt = float(opt["R2_ohm"])
                print(
                    f"[Auto-Tune] Result R1={R1_opt:.3f} ohm  "
                    f"R2={R2_opt:.3f} ohm",
                    flush=True,
                )
                if (abs(R1_opt - a_initial) < 1e-3
                        and abs(R2_opt - b_initial) < 1e-3):
                    QMessageBox.warning(
                        self, "Optimization Failed",
                        "Optimizasyon Başarısız:\n\n"
                        "Seçilen kapasitör değerleri ile standart 1.2/50 µs "
                        "dalgası elde edilebilecek uygun bir R1/R2 direnç "
                        "kombinasyonu bulunamadı."
                    )
                    return
                self.spinboxes["R1_ohm"].setValue(R1_opt)
                self.spinboxes["R2_ohm"].setValue(R2_opt)

            self.run_simulation()
            print("[Auto-Tune] Done.", flush=True)

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(
                self, "Auto-Tune Error",
                f"Could not optimize:\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "Full traceback printed to the terminal.",
            )
        finally:
            QApplication.restoreOverrideCursor()
            self.tune_btn.setText(self._tune_label_for(self._circuit_mode()))
            self.tune_btn.setEnabled(True)


def launch_gui(initial_config=None):
    """Create the QApplication and main window, then enter the event loop."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow(initial_config=initial_config)
    window.show()
    sys.exit(app.exec())
