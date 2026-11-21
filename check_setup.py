"""Run before deploying:  python check_setup.py
Reports which files are present, which model(s) the app will use, and whether versions match."""
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).parent
ok = True


def line(flag, msg):
    global ok
    ok = ok and flag
    print(("  OK       " if flag else "  MISSING  ") + msg)


def size_note(p):
    mb = p.stat().st_size / 1e6
    return f"{p.name} ({mb:.1f} MB)" + ("  <- over 25 MB: upload with git, not the web page" if mb > 25 else "")


print("Project files")
for name in ("app.py", "charts.py", "requirements.txt"):
    line((ROOT / name).is_file(), name)
line((ROOT / "assets").is_dir(), "assets/ (styling and scripts)")
line((ROOT / "data" / "augmented_balanced_dataset_m.csv").is_file(), "data/augmented_balanced_dataset_m.csv")

folders = sorted(p.parent for p in (ROOT / "models").glob("*/model.pkl"))
if not folders and (ROOT / "model.pkl").is_file():
    folders = [ROOT]
mode = "two models" if len(folders) >= 2 else "one model"
print(f"\nModels found: {len(folders)}  ->  app mode: {mode}")
if not folders:
    line(False, "model.pkl (project root, or models/<name>/model.pkl)")

kinds = []
for f in folders:
    rel = f.relative_to(ROOT) if f != ROOT else Path(".")
    print(f"\n[{rel}]")
    pkl, met = f / "model.pkl", f / "model_metrics.json"
    line(pkl.is_file(), size_note(pkl) if pkl.is_file() else "model.pkl")
    line(met.is_file(), "model_metrics.json")
    if not (pkl.is_file() and met.is_file()):
        continue
    M = json.loads(met.read_text())
    kind = "original data (has leak_check)" if "leak_check" in M else "deduplicated (has raw_split_comparison)"
    kinds.append("original" if "leak_check" in M else "dedup")
    print(f"           trained on: {kind}")
    print(f"           test accuracy: {M['test']['accuracy']:.1%} on {M['dataset']['test_rows']} records")
    try:
        import joblib
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            est = joblib.load(pkl)
        mism = [x for x in w if "version" in str(x.message).lower()]
        line(not mism, "scikit-learn version matches" if not mism else f"scikit-learn version mismatch: {mism[0].message}")
        print(f"           model: {type(est).__name__}, classes: {[str(c) for c in est.classes_]}")
    except Exception as e:  # noqa: BLE001
        line(False, f"model.pkl could not be loaded: {e}")

if mode == "two models" and sorted(set(kinds)) != ["dedup", "original"]:
    line(False, "two models found, but they are not one original-data and one deduplicated run")
orig_dirs = [f for f, k in zip(folders, kinds) if k == "original"] or folders[:1]
if orig_dirs:
    d = orig_dirs[0]
    has_imp = (d / "feature_importance.csv").is_file() or (ROOT / "feature_importance.csv").is_file()
    line(has_imp, "feature_importance.csv (next to the original-data model, or in the project root)")

try:
    import sklearn
    print(f"\nInstalled scikit-learn: {sklearn.__version__}  ->  requirements.txt line: scikit-learn=={sklearn.__version__}")
except ImportError:
    print("\nscikit-learn is not installed here")
print("\nREADY TO DEPLOY" if ok else "\nFIX THE ITEMS MARKED MISSING, THEN RUN AGAIN")
sys.exit(0 if ok else 1)
