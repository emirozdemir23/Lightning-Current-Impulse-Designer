"""One-shot project health check. Exits 0 only if every check passes."""
import sys, os, json, py_compile, importlib, inspect
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
failures = []

# ---- 1) Compile every Python file -----------------------------------------
py_files = []
for dirpath, _dirs, files in os.walk(ROOT):
    if "__pycache__" in dirpath:
        continue
    for f in files:
        if f.endswith(".py"):
            py_files.append(os.path.join(dirpath, f))

print("=== 1) Syntax / compile check ===")
for path in sorted(py_files):
    rel = os.path.relpath(path, ROOT)
    try:
        py_compile.compile(path, doraise=True)
        print(f"  OK   {rel}")
    except py_compile.PyCompileError as e:
        print(f"  FAIL {rel}: {e}")
        failures.append(f"compile:{rel}")

# ---- 2) config.json well-formed -------------------------------------------
print("\n=== 2) config.json validation ===")
try:
    with open(os.path.join(ROOT, "config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    gen = cfg["impulse_generator"]
    required = ["V0_kV", "C1_nF", "C2_nF", "R1_ohm", "R2_ohm", "L_uH",
                "circuit_mode"]
    missing = [k for k in required if k not in gen]
    if missing:
        print(f"  FAIL missing keys: {missing}")
        failures.append("config:missing")
    else:
        print(f"  OK   well-formed, keys present: {list(gen.keys())}")
except Exception as e:
    print(f"  FAIL {type(e).__name__}: {e}")
    failures.append("config:parse")
    gen = None

# ---- 3) Solver engine across all three modes ------------------------------
print("\n=== 3) Solver engine (no nan/inf) ===")
from src.solver import find_optimal_resistors, solve_impulse_circuit

def finite(name, arr):
    arr = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(arr)):
        bad = np.sum(~np.isfinite(arr))
        print(f"     FAIL {name}: {bad} non-finite value(s)")
        failures.append(f"solver:{name}")
        return False
    return True

base = dict(gen) if gen else {
    "V0_kV": 100.0, "C1_nF": 100.0, "C2_nF": 1.0,
    "R1_ohm": 500.0, "R2_ohm": 2500.0, "L_uH": 10.0, "circuit_mode": "RC"}
# Ensure C1 >> C2 so optimizer targets are physically reachable.
base["C1_nF"] = 100.0
base["C2_nF"] = 1.0

for mode in ("RC", "RLC", "Impulse Current"):
    print(f"  -- mode: {mode}")
    cfg_m = dict(base, circuit_mode=mode)
    try:
        opt = find_optimal_resistors(cfg_m)
        print(f"     find_optimal_resistors -> {opt}")
        finite(f"{mode}:opt", list(opt.values()))
        tuned = dict(cfg_m)
        tuned.update({k: v for k, v in opt.items()})
        t, v, i = solve_impulse_circuit(tuned)
        ok = all([
            finite(f"{mode}:t", t),
            finite(f"{mode}:V", v),
            finite(f"{mode}:i", i),
        ])
        print(f"     solve_impulse_circuit -> t[{len(t)}], "
              f"Vmax={np.max(np.abs(v)):.3f} kV, Imax={np.max(np.abs(i)):.4f} kA"
              f"  {'OK' if ok else 'FAIL'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        failures.append(f"solver:{mode}:exc")

# ---- 4) GUI logic: validation + tab-aware status bar ----------------------
print("\n=== 4) GUI logic checks ===")
gui_path = os.path.join(ROOT, "src", "gui.py")
with open(gui_path, encoding="utf-8") as fh:
    gui_src = fh.read()

# 4a) import the module (offscreen so no display needed)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    import src.gui as gui_mod
    print("  OK   src.gui imports cleanly")
except Exception as e:
    import traceback
    traceback.print_exc()
    failures.append("gui:import")
    gui_mod = None

# 4b) C1 >= 10*C2 validation present in run_simulation
if gui_mod is not None:
    run_src = inspect.getsource(gui_mod.MainWindow.run_simulation)
    if ("10 * config[\"C2_nF\"]" in run_src or "10*config" in run_src
            or "10 * config['C2_nF']" in run_src) and "QMessageBox.critical" in run_src:
        print("  OK   run_simulation guards C1 < 10*C2 with QMessageBox.critical")
    else:
        print("  FAIL run_simulation missing C1>=10*C2 validation")
        failures.append("gui:validation")
    # error message text
    if "en az 10 katı olmalıdır" in run_src:
        print("  OK   correct Turkish error message present")
    else:
        print("  FAIL validation error message text missing")
        failures.append("gui:msg")

    # 4c) tab-switch status bar logic
    has_update = hasattr(gui_mod.MainWindow, "_update_status_bar")
    if has_update:
        us_src = inspect.getsource(gui_mod.MainWindow._update_status_bar)
        wired = "currentChanged.connect" in gui_src and "_update_status_bar" in gui_src
        tab_aware = "currentIndex()" in us_src and "is_standard_8_20" in us_src \
            and "is_standard_lightning" in us_src
        if has_update and wired and tab_aware:
            print("  OK   _update_status_bar is tab-aware and wired to currentChanged")
        else:
            print(f"  FAIL status-bar logic incomplete "
                  f"(wired={wired}, tab_aware={tab_aware})")
            failures.append("gui:statusbar")
    else:
        print("  FAIL _update_status_bar method missing")
        failures.append("gui:statusbar")

# ---- Verdict ---------------------------------------------------------------
print("\n=== VERDICT ===")
if failures:
    print("FAILURES:", failures)
    sys.exit(1)
else:
    print("PROJECT READY FOR PRESENTATION - ALL SYSTEMS OK")
    sys.exit(0)
