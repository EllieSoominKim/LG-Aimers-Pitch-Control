"""Splice script_template_matchup_only.py + its embedded blobs into the
final script.py (the isolated matchup-feature single-change test)."""
import os

TRAINING_DIR = "C:/LG-Aimers-Pitch-Control/training"
OUT_PATH = "C:/LG-Aimers-Pitch-Control/script.py"
PLACEHOLDER = "__EMBEDDED_BLOBS_PLACEHOLDER__"

with open(os.path.join(TRAINING_DIR, "script_template_matchup_only.py"), "r", encoding="utf-8") as f:
    template = f.read()
with open(os.path.join(TRAINING_DIR, "_embedded_artifacts_snippet_matchup_only.py"), "r", encoding="utf-8") as f:
    blobs = f.read()

count = template.count(PLACEHOLDER)
assert count == 1, f"expected exactly 1 placeholder occurrence, found {count}"

final = template.replace(PLACEHOLDER, blobs)

with open(OUT_PATH, "w", encoding="utf-8", newline="\n") as f:
    f.write(final)

size = os.path.getsize(OUT_PATH)
print(f"Wrote {OUT_PATH} ({size:,} bytes)")

assert PLACEHOLDER not in final
for const in ("_EMBEDDED_ARTIFACTS_B64", "_EMBEDDED_MATCHUP_B64", "_EMBEDDED_LIGHTGBM_B64", "_EMBEDDED_HGB_B64", "_EMBEDDED_CATBOOST_B64"):
    n = final.count(const)
    assert n >= 2, f"{const} appears {n} times, expected >= 2 (definition + usage)"
    print(f"  {const}: appears {n} times")
print("Splice OK")
