"""Investigation item #2: instrument script.py's exact logic against the
real 5-row test.csv, printing which calibration branch (cold/warm) each
row takes, the raw model outputs, and the pre/post-calibration values --
verify branching threshold and calibrator selection match what was
validated offline in investigate_slice_calibration.py.
"""
import importlib.util
import sys

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control")
spec = importlib.util.spec_from_file_location("script_mod", "C:/LG-Aimers-Pitch-Control/script.py")
script_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_mod)  # NOTE: this runs main() only if __name__=="__main__", which it isn't here

import numpy as np

BASE_DIR = script_mod.BASE_DIR
MODEL_DIR = script_mod.__dict__.get("MODEL_DIR", "C:/LG-Aimers-Pitch-Control/model")
MODEL_DIR = "C:/LG-Aimers-Pitch-Control/model"
TEST_DIR = "C:/LG-Aimers-Pitch-Control/data"

categories = script_mod.load_categories_with_fallback(MODEL_DIR)
calibrators = script_mod.load_calibrators_with_fallback(MODEL_DIR)
stacker = script_mod.load_stacker_with_fallback(MODEL_DIR)
matchup_table = script_mod.load_matchup_table_with_fallback(MODEL_DIR)
ext_table = script_mod.load_matchup_extension_table_with_fallback(MODEL_DIR)

lgbm = script_mod.load_lightgbm_with_fallback(MODEL_DIR)
cb = script_mod.load_catboost_with_fallback(MODEL_DIR)
hgb = script_mod.load_hgb_with_fallback(MODEL_DIR)

test = script_mod.load_test(TEST_DIR + "/test.csv")
ids = test[script_mod.ID_COL].tolist()
X = script_mod.build_features(test, categories, matchup_table, ext_table)

print(f"COLD_THRESH = {script_mod.COLD_THRESH}")
print()
print("Per-row asof_pitcher_n and cold/warm branch:")
cold_mask = (X["asof_pitcher_n"] < script_mod.COLD_THRESH).values
for i, rid in enumerate(ids):
    branch = "COLD" if cold_mask[i] else "WARM"
    print(f"  {rid}: asof_pitcher_n={X['asof_pitcher_n'].values[i]:.1f}  -> {branch}")

p_lgbm_raw = lgbm.predict(X)
p_cb_raw = cb.predict_proba(script_mod.catboost_frame(X))[:, 1]
p_hgb_raw = hgb.predict_proba(script_mod.hgb_frame(X))[:, 1]

print()
print("Raw model outputs (pre-calibration):")
for i, rid in enumerate(ids):
    print(f"  {rid}: lgbm={p_lgbm_raw[i]:.5f}  cb={p_cb_raw[i]:.5f}  hgb={p_hgb_raw[i]:.5f}")

p_lgbm_cal = script_mod.apply_segmented_calibration(p_lgbm_raw, calibrators["lightgbm"], cold_mask)
p_cb_cal = script_mod.apply_segmented_calibration(p_cb_raw, calibrators["catboost"], cold_mask)
p_hgb_cal = script_mod.apply_segmented_calibration(p_hgb_raw, calibrators["hgb"], cold_mask)

print()
print("Calibrated (post-segmented-calibration) outputs:")
for i, rid in enumerate(ids):
    branch = "COLD" if cold_mask[i] else "WARM"
    print(f"  {rid} [{branch}]: lgbm {p_lgbm_raw[i]:.5f}->{p_lgbm_cal[i]:.5f}  "
          f"cb {p_cb_raw[i]:.5f}->{p_cb_cal[i]:.5f}  hgb {p_hgb_raw[i]:.5f}->{p_hgb_cal[i]:.5f}")

# Manually verify the calibrator arrays themselves: print a few threshold
# points from each model's cold vs warm calibrator to eyeball if they're
# sane (monotonic, reasonable range) and genuinely different from each other.
print()
print("Calibrator threshold arrays (first/last 3 points each):")
for name in ("lightgbm", "catboost", "hgb"):
    c = calibrators[name]
    # c["cold"]/c["warm"] are closures; recover their bound x/y via inspection
    for branch in ("cold", "warm"):
        fn = c[branch]
        closure_vars = dict(zip(fn.__code__.co_freevars, (cv.cell_contents for cv in fn.__closure__)))
        x, y = closure_vars.get("x"), closure_vars.get("y")
        if x is not None:
            print(f"  {name}/{branch}: n_points={len(x)}  x[:3]={x[:3]}  x[-3:]={x[-3:]}  "
                  f"y[:3]={y[:3]}  y[-3:]={y[-3:]}")

preds = script_mod.stacker_blend({"lightgbm": p_lgbm_cal, "catboost": p_cb_cal, "hgb": p_hgb_cal}, stacker)
print()
print("Final stacked predictions:")
for i, rid in enumerate(ids):
    print(f"  {rid}: {preds[i]:.5f}")
