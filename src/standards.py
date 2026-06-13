"""IEC 60060-1 impulse-waveform compliance evaluation.

Two evaluators are exposed:

* ``evaluate_lightning_impulse(t, v)`` — voltage 1.2/50 μs standard.
      T1 = 1.67 * (t_90% - t_30%);  t0 = t_30% - 0.3 * T1;
      T2 = t_50%_tail - t0
      Pass: T1 ∈ 1.2 μs ± 30 %,  T2 ∈ 50 μs ± 20 %.

* ``evaluate_current_impulse(t, i)`` — current 8/20 μs standard.
      T1 = 1.25 * (t_90% - t_10%);  t0 = t_10% - 0.1 * T1;
      T2 = t_50%_tail - t0
      Pass: T1 ∈ 8 μs ± 10 %,  T2 ∈ 20 μs ± 10 %.
"""

import numpy as np

_T1_NOM_US = 1.2
_T1_TOL = 0.30        # ±30 %
_T2_NOM_US = 50.0
_T2_TOL = 0.20        # ±20 %


def _interp_crossing(t, v, target, lo, hi):
    """Return the first time in samples [lo, hi] where v crosses `target`.

    Detects a sign change of (v - target) across consecutive samples and
    linearly interpolates between them. Returns None if no crossing exists
    in the segment.
    """
    for i in range(lo, hi):
        v0, v1 = v[i], v[i + 1]
        if v0 == v1:
            continue
        if (v0 - target) * (v1 - target) <= 0.0:
            return t[i] + (target - v0) * (t[i + 1] - t[i]) / (v1 - v0)
    return None


def evaluate_lightning_impulse(t, v):
    """Compute Vmax, T1, T2, and the IEC 60060-1 lightning-impulse pass flag.

    Parameters
    ----------
    t : array-like
        Monotonically increasing time samples, in microseconds.
    v : array-like
        Output impulse voltage samples, in kV, sampled at ``t``.

    Returns
    -------
    dict
        ``{"Vmax_kV": float,
            "T1_us":   float,
            "T2_us":   float,
            "is_standard_lightning": bool}``

        ``is_standard_lightning`` is True iff T1 ∈ 1.2 μs ± 30 % and
        T2 ∈ 50 μs ± 20 %. T1/T2 are NaN if the required threshold
        crossings cannot be located on the provided trace.
    """
    t = np.asarray(t, dtype=float)
    v = np.asarray(v, dtype=float)

    peak_idx = int(np.argmax(v))
    Vmax = float(v[peak_idx])

    t30 = _interp_crossing(t, v, 0.30 * Vmax, 0, peak_idx)
    t90 = _interp_crossing(t, v, 0.90 * Vmax, 0, peak_idx)
    t50_tail = _interp_crossing(t, v, 0.50 * Vmax, peak_idx, len(v) - 1)

    if t30 is None or t90 is None:
        T1 = float("nan")
        t0 = float("nan")
    else:
        T1 = 1.67 * (t90 - t30)
        t0 = t30 - 0.3 * T1

    if t50_tail is None or np.isnan(t0):
        T2 = float("nan")
    else:
        T2 = t50_tail - t0

    T1_ok = (
        not np.isnan(T1)
        and _T1_NOM_US * (1 - _T1_TOL) <= T1 <= _T1_NOM_US * (1 + _T1_TOL)
    )
    T2_ok = (
        not np.isnan(T2)
        and _T2_NOM_US * (1 - _T2_TOL) <= T2 <= _T2_NOM_US * (1 + _T2_TOL)
    )

    return {
        "Vmax_kV": Vmax,
        "T1_us": float(T1),
        "T2_us": float(T2),
        "is_standard_lightning": bool(T1_ok and T2_ok),
    }


_I_T1_NOM_US = 8.0
_I_T1_TOL = 0.10        # ±10 %
_I_T2_NOM_US = 20.0
_I_T2_TOL = 0.10        # ±10 %


def evaluate_current_impulse(t, i):
    """Compute Imax, T1, T2, and the IEC 60060-1 Current 8/20 pass flag.

    Definitions (IEC 60060-1 impulse current):
        T1 = 1.25 * (t_90% - t_10%)              # front time
        t0  = t_10% - 0.1 * T1                   # virtual origin
        T2 = t_50%_tail - t0                     # tail time to half-value

    Pass when T1 ∈ 8.0 μs ± 10 % and T2 ∈ 20.0 μs ± 10 %.
    """
    t = np.asarray(t, dtype=float)
    i = np.asarray(i, dtype=float)

    peak_idx = int(np.argmax(i))
    Imax = float(i[peak_idx])

    t10 = _interp_crossing(t, i, 0.10 * Imax, 0, peak_idx)
    t90 = _interp_crossing(t, i, 0.90 * Imax, 0, peak_idx)
    t50_tail = _interp_crossing(t, i, 0.50 * Imax, peak_idx, len(i) - 1)

    if t10 is None or t90 is None:
        T1 = float("nan")
        t0 = float("nan")
    else:
        T1 = 1.25 * (t90 - t10)
        t0 = t10 - 0.1 * T1

    if t50_tail is None or np.isnan(t0):
        T2 = float("nan")
    else:
        T2 = t50_tail - t0

    T1_ok = (
        not np.isnan(T1)
        and _I_T1_NOM_US * (1 - _I_T1_TOL) <= T1 <= _I_T1_NOM_US * (1 + _I_T1_TOL)
    )
    T2_ok = (
        not np.isnan(T2)
        and _I_T2_NOM_US * (1 - _I_T2_TOL) <= T2 <= _I_T2_NOM_US * (1 + _I_T2_TOL)
    )

    return {
        "Imax_kA": Imax,
        "T1_us": float(T1),
        "T2_us": float(T2),
        "is_standard_8_20": bool(T1_ok and T2_ok),
    }
