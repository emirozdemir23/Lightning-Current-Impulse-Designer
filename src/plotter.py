"""Visualization for the high-voltage impulse response.

Two entry points:

* ``draw_impulse_response(ax, t, V_C1, V_C2, evaluation)``
    Pure drawing on a caller-owned Axes — used by the file-output
    wrapper *and* by the PyQt6 GUI canvas. The function clears the
    Axes first, so it is safe to call repeatedly for live redraws.

* ``plot_impulse_response(t, V_C1, V_C2, evaluation, save_path=...)``
    Convenience wrapper that builds its own headless Figure / Agg
    canvas, draws onto it, and saves a PNG. It does NOT touch
    matplotlib.pyplot or the global backend, so it is safe to call
    from a process that has already initialised a Qt backend.
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def draw_impulse_response(ax, t, V_C1, V_C2, evaluation):
    """Render the impulse-response visualization onto an existing Axes.

    Auto-detects voltage-mode (key ``Vmax_kV``) vs current-mode
    (key ``Imax_kA``) from the evaluation dict and adapts curve set,
    reference fractions, y-axis label, title, and metrics box.
    """
    ax.clear()
    is_current = "Imax_kA" in evaluation

    if is_current:
        peak = evaluation["Imax_kA"]
        std_pass = evaluation.get("is_standard_8_20", False)
        ref_levels = [
            (1.0, r"$I_{max}$"),
            (0.9, r"$0.9\,I_{max}$"),
            (0.5, r"$0.5\,I_{max}$"),
            (0.1, r"$0.1\,I_{max}$"),
        ]
        peak_text = f"$I_{{max}}$ = {peak:.2f} kA"
        std_label = "8/20"
        y_label = "Current (kA)"
        title = "Impulse Generator Response — IEC 60060-2 8/20 µs Current Wave"
        ax.plot(t, V_C2, color="#c44536", linewidth=2.0,
                label=r"$i_L(t)$ — output current")
    else:
        peak = evaluation["Vmax_kV"]
        std_pass = evaluation.get("is_standard_lightning", False)
        ref_levels = [
            (1.0, r"$V_{max}$"),
            (0.9, r"$0.9\,V_{max}$"),
            (0.5, r"$0.5\,V_{max}$"),
            (0.3, r"$0.3\,V_{max}$"),
        ]
        peak_text = f"$V_{{max}}$ = {peak:.2f} kV"
        std_label = "1.2/50"
        y_label = "Voltage (kV)"
        title = "Impulse Generator Response — IEC 60060-1 Lightning Evaluation"
        ax.plot(t, V_C1, color="#4a6fa5", linewidth=1.4, linestyle="--",
                alpha=0.85, label=r"$V_{C1}(t)$ — storage cap")
        ax.plot(t, V_C2, color="#c44536", linewidth=2.0,
                label=r"$V_{C2}(t)$ — output impulse")

    for frac, sym in ref_levels:
        level = frac * peak
        ax.axhline(level, color="gray", linestyle=":",
                   linewidth=0.9, alpha=0.7)
        ax.text(
            t[-1] * 0.985, level, sym,
            va="center", ha="right",
            fontsize=8, color="dimgray",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="white", edgecolor="none", alpha=0.85),
        )

    pass_label = "PASS" if std_pass else "FAIL"
    metrics = (
        f"{peak_text}\n"
        f"$T_1$ = {evaluation['T1_us']:.3f} μs\n"
        f"$T_2$ = {evaluation['T2_us']:.2f} μs\n"
        f"IEC {std_label}: {pass_label}"
    )
    ax.text(
        0.97, 0.97, metrics, transform=ax.transAxes,
        ha="right", va="top", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5",
                  facecolor="white", edgecolor="gray", alpha=0.92),
    )

    ax.set_xlabel("Time (μs)")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right")


def plot_impulse_response(t, V_C1, V_C2, evaluation,
                          save_path="impulse_plot.png"):
    """Render to a fresh headless figure and save it as a PNG."""
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)        # bind an Agg canvas locally; no global state
    ax = fig.add_subplot(111)
    draw_impulse_response(ax, t, V_C1, V_C2, evaluation)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    return save_path
