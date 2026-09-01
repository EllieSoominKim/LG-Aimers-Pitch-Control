"""Rebuild submit.zip for the Phase-1-validated matchup+extensions
candidate. Same file list as matchup-only plus matchup_extension_lookup.json.
"""
import os
import zipfile

ROOT = "C:/LG-Aimers-Pitch-Control"
OUT = os.path.join(ROOT, "submit.zip")

FILES = [
    "script.py",
    "requirements.txt",
    "model/calibration_catboost.npz",
    "model/calibration_hgb.npz",
    "model/calibration_lightgbm.npz",
    "model/catboost.cbm",
    "model/categories.json",
    "model/hgb.joblib",
    "model/lightgbm.txt",
    "model/matchup_lookup.json",
    "model/matchup_extension_lookup.json",
]

if os.path.exists(OUT):
    os.remove(OUT)

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for rel in FILES:
        src = os.path.join(ROOT, rel.replace("/", os.sep))
        assert os.path.isfile(src), f"missing: {src}"
        zf.write(src, arcname=rel)

size = os.path.getsize(OUT)
print(f"Wrote {OUT} ({size:,} bytes)")

with zipfile.ZipFile(OUT) as zf:
    print(f"\n{'name':40s} {'size':>12s}")
    for info in zf.infolist():
        print(f"{info.filename:40s} {info.file_size:>12,}")
    bad = zf.testzip()
    print(f"\ntestzip() bad file: {bad!r}  (None = all entries OK)")
