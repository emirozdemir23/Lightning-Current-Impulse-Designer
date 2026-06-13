"""ODE solver for the two-mesh impulse generator.

The ``circuit_mode`` key in the (flat) config dict selects the topology:

* ``"RC"`` — 2-state system. State vector ``(V_C1, V_C2)``:

      node1 ---[R1]--- node2
        |               |
       C1              C2  (R2 in parallel with C2)
        |               |
       gnd             gnd

      dV1/dt = -(V1 - V2) / (R1 * C1)
      dV2/dt = ((V1 - V2) / R1 - V2 / R2) / C2

* ``"RLC"`` / ``"Impulse Current"`` — 3-state system; inserts a stray
  series inductance L between R1 and node 2:

      node1 ---[R1]---[L]--- node2
        |                     |
       C1                    C2  (R2 in parallel with C2)

      dV1/dt  = -i_L / C1
      dV2/dt  = (i_L - V2 / R2) / C2
      di_L/dt = (V1 - V2 - i_L * R1) / L

Integration runs in SI internally. ``solve_impulse_circuit`` returns
the unified 3-tuple ``(t_us, V_C2_kV, i_kA)`` for every mode: V_C2 is
the output (load) voltage; i is the loop current — in RC mode it is
reconstructed as ``(V_C1 - V_C2) / R1``, in RLC / Impulse-Current
modes it is the integrated state variable ``i_L``.
"""

import copy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from src.standards import evaluate_lightning_impulse, evaluate_current_impulse


def solve_impulse_circuit(config, t_max_us=200, points=2000,
                          rtol=1e-8, atol=1e-10):
    """Integrate the impulse-generator state-space ODEs.

    The flat config dict carries ``V0_kV``, ``C1_nF``, ``C2_nF``,
    ``R1_ohm``, ``R2_ohm``, ``L_uH`` and ``circuit_mode`` (one of
    ``"RC"``, ``"RLC"``, or ``"Impulse Current"``). The integration
    horizon ``t_max_us`` is auto-expanded based on the dominant tail
    time-constant so the 50 %-tail crossing falls inside the window
    even for large R2·C1 products or long L/R1 transients.

    ``points`` and the ``rtol`` / ``atol`` solver tolerances are exposed
    so callers can trade accuracy for speed: the auto-tune optimizer
    runs thousands of these solves and passes a coarse, fast setting for
    its intermediate cost evaluations, while the final post-optimization
    plot keeps the tight defaults (full precision).

    Returns
    -------
    t_us : ndarray, shape (points,)
        Time samples in microseconds.
    V_C2_kV : ndarray, shape (points,)
        Output (load) capacitor voltage in kV.
    i_kA : ndarray, shape (points,)
        Loop current in kiloamperes (RC: derived from
        ``(V_C1 − V_C2) / R1``; RLC / Impulse Current: integrated
        state variable ``i_L`` rescaled to kA).
    """
    V0 = config["V0_kV"] * 1e3        # kV -> V
    C1 = config["C1_nF"] * 1e-9       # nF -> F
    C2 = config["C2_nF"] * 1e-9       # nF -> F
    R1 = float(config["R1_ohm"])      # Ω
    R2 = float(config["R2_ohm"])      # Ω

    circuit_mode = config.get("circuit_mode", "RC")

    # Defensive guards for the programmatic API (the GUI's spinbox
    # floors already enforce these in interactive use). C2/R2 are
    # checked only for the modes that physically depend on the load
    # branch — Impulse Current grays them out in the UI.
    assert C1 > 0.0, "C1_nF must be > 0"
    assert R1 > 0.0, "R1_ohm must be > 0"
    if circuit_mode != "Impulse Current":
        assert C2 > 0.0, "C2_nF must be > 0"
        assert R2 > 0.0, "R2_ohm must be > 0"

    # Dynamic horizon: ensure the 50 %-tail crossing falls inside the
    # integration window for large R2·C1 products.
    tau_tail_us = float(config["R2_ohm"]) * float(config["C1_nF"]) * 0.001
    if circuit_mode == "Impulse Current":
        # R2 is NOT in the current-discharge branch (grayed out in the
        # UI), so its R2·C1 tail term must not inflate the horizon —
        # leaving it in dragged t_max into the millisecond range and made
        # every stiff Radau solve crawl. The 8/20 µs current is fully
        # captured by the 200 µs default floor; the only physical tail
        # here is the L/R1 transient, so clamp to that plus the floor.
        L_uH_val = float(config.get("L_uH", 0.0))
        t_max_us = max(
            float(t_max_us),
            50.0 * L_uH_val / max(1.0, R1),
        )
    else:
        t_max_us = max(float(t_max_us), 7.0 * tau_tail_us)

    if circuit_mode in ("RLC", "Impulse Current"):
        L = config.get("L_uH", 0.0) * 1e-6     # μH -> H
        if L <= 0.0:
            raise ValueError(
                f"{circuit_mode} mode requires a strictly positive L_uH "
                "(stray inductance)."
            )

        def rhs(_t, y):
            v1, v2, iL = y
            return [
                -iL / C1,
                (iL - v2 / R2) / C2,
                (v1 - v2 - iL * R1) / L,
            ]
        y0 = [V0, 0.0, 0.0]

    elif circuit_mode == "RC":
        def rhs(_t, y):
            v1, v2 = y
            i_front = (v1 - v2) / R1
            return [-i_front / C1, (i_front - v2 / R2) / C2]
        y0 = [V0, 0.0]

    else:
        raise ValueError(
            f"Unknown circuit_mode {circuit_mode!r}; "
            "expected 'RC', 'RLC', or 'Impulse Current'."
        )

    t_max_s = t_max_us * 1e-6
    t_eval = np.linspace(0.0, t_max_s, points)

    # The 8/20 µs current discharge is a highly stiff transient (large
    # L/R1 di/dt against a fast C1 swing); an explicit RK45 step shrinks
    # to keep up and can stall the integrator, freezing the GUI. Use the
    # implicit Radau method there, which steps stiff systems stably.
    method = "Radau" if circuit_mode == "Impulse Current" else "RK45"

    sol = solve_ivp(
        rhs,
        t_span=(0.0, t_max_s),
        y0=y0,
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"solve_ivp failed: {sol.message}")

    t_us = sol.t * 1e6
    V_C2_kV = sol.y[1] / 1e3
    if circuit_mode == "RC":
        # Loop current through R1 (= the current that charges C2 and
        # bleeds R2). kV / Ω = kA, so no extra unit conversion needed.
        i_kA = (sol.y[0] / 1e3 - V_C2_kV) / R1
    else:
        # RLC and Impulse Current expose i_L as state index 2 in
        # amperes; convert to kA for the unified contract.
        i_kA = sol.y[2] / 1e3
    return t_us, V_C2_kV, i_kA


def _analytical_rc_resistors(config, target_T1, target_T2):
    """Closed-form R1, R2 for the RC double-exponential generator.

    Inverts the classic high-voltage impulse approximations — exact in
    the C1 ≫ C2 regime the 1.2/50 front network is built for — so no ODE
    iteration is needed::

        T1 ≈ 3 · R1 · (C1·C2)/(C1 + C2)     (front / rise time)
        T2 ≈ 0.7 · (R1 + R2) · (C1 + C2)    (tail / half-value time)

    Solving for the resistors::

        R1 = T1 / (3 · C_series),   C_series = (C1·C2)/(C1 + C2)
        R2 = T2 / (0.7 · (C1 + C2)) − R1

    Everything is computed in SI (F, s, Ω). Both resistors are floored
    at 0.1 Ω so a degenerate target can never hand the GUI a
    non-positive value.

    Returns
    -------
    dict
        ``{"R1_ohm": float, "R2_ohm": float}``.
    """
    C1 = config["C1_nF"] * 1e-9       # nF -> F
    C2 = config["C2_nF"] * 1e-9       # nF -> F
    T1 = target_T1 * 1e-6             # μs -> s
    T2 = target_T2 * 1e-6             # μs -> s

    C_series = (C1 * C2) / (C1 + C2)
    R1 = T1 / (3.0 * C_series)
    R2 = T2 / (0.7 * (C1 + C2)) - R1
    return {"R1_ohm": float(max(R1, 0.1)), "R2_ohm": float(max(R2, 0.1))}


def find_optimal_resistors(config, target_T1=1.2, target_T2=50.0):
    """Optimize the active mode's tuning parameters to a target wave.

    Inspects ``config["circuit_mode"]`` from the flat config dict and
    chooses the tuning strategy accordingly:

    * ``"RC"`` mode solves for ``[R1, R2]`` **analytically** via
      :func:`_analytical_rc_resistors`, inverting the standard 1.2/50 μs
      impulse equations for an instant, iteration-free result.
    * ``"RLC"`` mode tunes ``[R1, R2]`` against the 1.2/50 μs lightning
      impulse target with the Nelder-Mead simplex (the stray inductor
      breaks the closed-form RC relations, so the ODE is still needed).
    * ``"Impulse Current"`` mode tunes ``[R1, L_uH]`` against the
      8/20 μs current impulse target (hard-coded 8.0 / 20.0 inside
      the cost function — ``target_T1`` / ``target_T2`` are ignored).

    The iterative modes use ``scipy.optimize.minimize`` with the
    Nelder-Mead simplex, capped at ``maxiter=500``. A smooth lower-bound
    clamp combined with a progressive distance penalty give the simplex
    a directional gradient out of infeasible regions instead of a flat
    penalty wall that traps the simplex.

    Parameters
    ----------
    config : dict
        Flat config dict (V0_kV, C1_nF, C2_nF, R1_ohm, R2_ohm, L_uH,
        circuit_mode). The active mode's tuning parameters seed the
        Nelder-Mead initial simplex (iterative modes only).
    target_T1, target_T2 : float
        Front and tail times in microseconds for voltage modes
        (defaults: IEC 60060-1 lightning 1.2 / 50 μs). Ignored in
        Impulse Current mode.

    Returns
    -------
    dict
        Voltage modes: ``{"R1_ohm": float, "R2_ohm": float}``.
        Impulse Current mode: ``{"R1_ohm": float, "L_uH": float}``.
    """
    base = copy.deepcopy(config)
    mode = base.get("circuit_mode", "RC")

    # RC mode has a closed-form inverse — skip the ODE solver entirely.
    if mode == "RC":
        return _analytical_rc_resistors(base, target_T1, target_T2)

    is_current = mode == "Impulse Current"

    if is_current:
        x0 = np.array([float(base["R1_ohm"]), float(base["L_uH"])])
    else:
        x0 = np.array([float(base["R1_ohm"]), float(base["R2_ohm"])])

    # Returned for infeasible / unevaluatable points. Acts as a soft
    # wall for Nelder-Mead (which has no native bound support).
    PENALTY = 1.0e6

    def cost(x):
        # Pump the Qt event loop each iteration so the GUI keeps
        # repainting and stays responsive during the optimization rather
        # than appearing frozen. No-op when run headless / without a live
        # QApplication (standalone scripts, tests).
        try:
            from PyQt6.QtWidgets import QApplication
            if QApplication.instance() is not None:
                QApplication.processEvents()
        except Exception:
            pass

        if is_current:
            # Impulse Current: x = [R1, L_uH]; target 8/20 µs current.
            R1, L = x
            R1_sim = max(0.1, R1)
            L_sim = max(0.1, L)
            penalty = 0.0
            if R1 < 0.1:
                penalty += (0.1 - R1) * 1000.0
            if L < 0.1:
                penalty += (0.1 - L) * 1000.0
            base["R1_ohm"] = float(R1_sim)
            base["L_uH"] = float(L_sim)
            try:
                # Coarse, fast solve for the intermediate cost: fewer
                # points and a loose tolerance are plenty to locate T1/T2
                # for the simplex. The winning point is re-solved at full
                # precision by the GUI's post-tune run_simulation().
                t, _, current = solve_impulse_circuit(
                    base, points=600, rtol=1e-3, atol=1e-6)
                ev = evaluate_current_impulse(t, current)
            except Exception:
                return PENALTY
            T1, T2 = ev["T1_us"], ev["T2_us"]
            if not (np.isfinite(T1) and np.isfinite(T2)):
                return PENALTY
            return (T1 - 8.0) ** 2 + (T2 - 20.0) ** 2 + penalty

        # Voltage modes: x = [R1, R2]; smooth clamp + progressive
        # penalty so Nelder-Mead has a gradient out of the invalid
        # region instead of a flat 1e6 wall that traps the simplex.
        R1, R2 = x
        R1_sim = max(1.0, R1)
        R2_sim = max(R1_sim + 1.0, R2)
        penalty = 0.0
        if R1 < 1.0:
            penalty += (1.0 - R1) * 1000.0
        if R2 < R1:
            penalty += (R1 - R2) * 1000.0
        base["R1_ohm"] = float(R1_sim)
        base["R2_ohm"] = float(R2_sim)
        try:
            t, V_C2, _ = solve_impulse_circuit(base)
            ev = evaluate_lightning_impulse(t, V_C2)
        except Exception:
            return PENALTY
        T1, T2 = ev["T1_us"], ev["T2_us"]
        if not (np.isfinite(T1) and np.isfinite(T2)):
            return PENALTY
        return (T1 - target_T1) ** 2 + (T2 - target_T2) ** 2 + penalty

    # Impulse Current optimization runs an expensive stiff ODE solve per
    # cost evaluation, so cap the simplex hard (maxiter=40) and loosen the
    # convergence tolerance (tol=1e-2). This trades a little precision for
    # a bounded, snappy run that can't spiral into thousands of sub-loops
    # and lock up the GUI. Voltage (RLC) tuning keeps the wider budget.
    if is_current:
        result = minimize(
            cost, x0,
            method="Nelder-Mead",
            tol=1e-2,
            options={"maxiter": 40},
        )
    else:
        result = minimize(
            cost, x0,
            method="Nelder-Mead",
            options={"maxiter": 500},
        )

    if is_current:
        R1_opt, L_opt = result.x
        return {"R1_ohm": float(R1_opt), "L_uH": float(L_opt)}
    R1_opt, R2_opt = result.x
    return {"R1_ohm": float(R1_opt), "R2_ohm": float(R2_opt)}
