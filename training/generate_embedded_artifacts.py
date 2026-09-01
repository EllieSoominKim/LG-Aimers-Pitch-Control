"""Generate self-contained base64 blobs of every model/ artifact to embed
directly inside script.py's source code, eliminating any dependency on
model/ file-staging surviving on the eval server.

History: a submission crashed with FileNotFoundError on model/categories.json,
which we could not reproduce locally (byte-for-byte verified: no casing/
naming bug in the zip or script.py). We embedded categories.json +
calibration breakpoints directly in script.py as a fix. The NEXT submission
then failed on model/lightgbm.txt -- the very next file main() tries to
load after the (now-fixed) categories/calibration block -- with the zip's
copy independently verified healthy (byte-identical to the standalone file,
loads fine locally). That pattern (each subsequent artifact fails in turn
as the previous one gets patched) points to a systematic issue with model/
file-staging on the eval server, not corruption. So: embed everything.

- categories.json + calibration_*.npz: bundled into one small gzip+base64
  JSON blob (as before, ~10KB raw).
- lightgbm.txt, hgb.joblib, catboost.cbm: each gzip-compressed and
  base64-encoded independently (no JSON wrapping needed, they're already
  serialized binary/text formats) -- avoids double-base64 overhead.

All three model formats support loading straight from in-memory bytes (no
temp file needed): lgb.Booster(model_str=...), joblib.load(io.BytesIO(...)),
CatBoostClassifier().load_model(blob=...).
"""

import base64
import gzip
import json
import os

MODEL_DIR = "C:/LG-Aimers-Pitch-Control/model"
OUT_DIR = "C:/LG-Aimers-Pitch-Control/training"

# ---------------------------------------------------------------------
# Small auxiliary data: categories.json + calibration_*.npz + stacker.json
# -> one blob
# ---------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "categories.json"), "r", encoding="utf-8") as f:
    categories = json.load(f)
with open(os.path.join(MODEL_DIR, "stacker.json"), "r", encoding="utf-8") as f:
    stacker = json.load(f)

import numpy as np
calibration = {}
for name in ("lightgbm", "catboost", "hgb"):
    npz = np.load(os.path.join(MODEL_DIR, f"calibration_{name}.npz"))
    calibration[name] = {
        "x_cold": npz["x_cold"].tolist(), "y_cold": npz["y_cold"].tolist(),
        "x_warm": npz["x_warm"].tolist(), "y_warm": npz["y_warm"].tolist(),
    }

aux_payload = {"categories": categories, "calibration": calibration, "stacker": stacker}
aux_json = json.dumps(aux_payload, separators=(",", ":")).encode("utf-8")
aux_b64 = base64.b64encode(gzip.compress(aux_json, compresslevel=9, mtime=0)).decode("ascii")
print(f"aux (categories+calibration+stacker): raw={len(aux_json):,}  b64={len(aux_b64):,}")

# ---------------------------------------------------------------------
# Matchup lookup table + round-4 extension lookup table: each its own blob
# (JSON, compresses well)
# ---------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "matchup_lookup.json"), "r", encoding="utf-8") as f:
    matchup_raw = f.read()
matchup_b64 = base64.b64encode(gzip.compress(matchup_raw.encode("utf-8"), compresslevel=9, mtime=0)).decode("ascii")
print(f"matchup_lookup.json: raw={len(matchup_raw):,}  b64={len(matchup_b64):,}")

with open(os.path.join(MODEL_DIR, "matchup_extension_lookup.json"), "r", encoding="utf-8") as f:
    matchup_ext_raw = f.read()
matchup_ext_b64 = base64.b64encode(gzip.compress(matchup_ext_raw.encode("utf-8"), compresslevel=9, mtime=0)).decode("ascii")
print(f"matchup_extension_lookup.json: raw={len(matchup_ext_raw):,}  b64={len(matchup_ext_b64):,}")

# ---------------------------------------------------------------------
# Large model files: each its own independent gzip+base64 blob
# ---------------------------------------------------------------------
model_blobs = {}
for const_name, fname in [
    ("LIGHTGBM", "lightgbm.txt"),
    ("HGB", "hgb.joblib"),
    ("CATBOOST", "catboost.cbm"),
]:
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(gzip.compress(raw, compresslevel=6, mtime=0)).decode("ascii")
    model_blobs[const_name] = b64
    print(f"{fname}: raw={len(raw):,}  b64={len(b64):,}")


def format_blob(var_name, b64, width=100):
    lines = [b64[i:i + width] for i in range(0, len(b64), width)]
    body = "\n".join(f'    "{line}"' for line in lines)
    return f"{var_name} = (\n{body}\n)\n"


out_path = os.path.join(OUT_DIR, "_embedded_artifacts_snippet.py")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(format_blob("_EMBEDDED_ARTIFACTS_B64", aux_b64))
    f.write("\n")
    f.write(format_blob("_EMBEDDED_MATCHUP_B64", matchup_b64))
    f.write("\n")
    f.write(format_blob("_EMBEDDED_MATCHUP_EXT_B64", matchup_ext_b64))
    f.write("\n")
    f.write(format_blob("_EMBEDDED_LIGHTGBM_B64", model_blobs["LIGHTGBM"]))
    f.write("\n")
    f.write(format_blob("_EMBEDDED_HGB_B64", model_blobs["HGB"]))
    f.write("\n")
    f.write(format_blob("_EMBEDDED_CATBOOST_B64", model_blobs["CATBOOST"]))

total_size = os.path.getsize(out_path)
print(f"\nWrote {out_path} ({total_size:,} bytes)")

# Round-trip sanity checks
decoded_aux = json.loads(gzip.decompress(base64.b64decode(aux_b64)))
assert decoded_aux == aux_payload
decoded_matchup = gzip.decompress(base64.b64decode(matchup_b64)).decode("utf-8")
assert json.loads(decoded_matchup) == json.loads(matchup_raw)
decoded_matchup_ext = gzip.decompress(base64.b64decode(matchup_ext_b64)).decode("utf-8")
assert json.loads(decoded_matchup_ext) == json.loads(matchup_ext_raw)
for const_name, fname in [("LIGHTGBM", "lightgbm.txt"), ("HGB", "hgb.joblib"), ("CATBOOST", "catboost.cbm")]:
    with open(os.path.join(MODEL_DIR, fname), "rb") as f:
        expected = f.read()
    decoded = gzip.decompress(base64.b64decode(model_blobs[const_name]))
    assert decoded == expected, f"{fname} round-trip mismatch!"
print("All round-trip checks passed.")
