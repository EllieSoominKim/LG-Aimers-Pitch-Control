"""Round 3, item #5: does calibration quality differ across cold-start vs.
warm pitchers, or across leverage (li) buckets? If a segmented calibrator
(fit separately per segment) beats the single global calibrator on VAL,
adopt it; otherwise the added complexity isn't worth it.

Approach: for each of the (already-tuned, matchup-feature) base models,
fit BOTH a global isotonic calibrator (as today) and per-segment
calibrators (fit only on that segment's CAL rows), then compare BSS on the
corresponding VAL segment for each. Uses whichever tuned params are
available (round 3 if present, else round 2) so this can run independently
of the Optuna searches finishing.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

d = load_split(include_matchup_extensions=True)  # item #4 confirmed extensions help (+0.00134/+0.00041)
X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
X_cal, y_cal = d["X_cal"], d["y_cal"]
X_val, y_val = d["X_val"], d["y_val"]
p_clim = d["p_clim"]


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


def load_params(paths, default):
    for path in paths:
        try:
            with open(path) as f:
                p = json.load(f)
            for k_ in list(p.keys()):
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "best_iteration"):
                    p.pop(k_, None)
            print(f"Loaded params from {path}")
            return p
        except FileNotFoundError:
            continue
    print("No tuned params found, using default")
    return default


lgbm_params = load_params(
    ["C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json",
     "C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params.json"],
    dict(decay=0.07, num_leaves=72, min_child_samples=1620, learning_rate=0.008,
         reg_lambda=0.012, reg_alpha=0.093, subsample=0.51, colsample_bytree=0.57))
cb_params = load_params(
    ["C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v2.json",
     "C:/LG-Aimers-Pitch-Control/training/best_catboost_params.json"],
    dict(decay=0.233, depth=7, l2_leaf_reg=28.8, min_data_in_leaf=131, learning_rate=0.035,
         random_strength=3.79, bagging_temperature=4.32))

print("\n=== Training LightGBM ===")
t0 = time.time()
lp = dict(lgbm_params)
decay_lgbm = lp.pop("decay")
w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
lgbm = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **lp)
lgbm.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
         eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
         callbacks=[lgb.early_stopping(80, verbose=False)])
print(f"  best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")

print("\n=== Training CatBoost ===")
t0 = time.time()
cp = dict(cb_params)
decay_cb = cp.pop("decay")
w_tr_cb = decay_cb ** (2024 - season_tr)
X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                         cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                         thread_count=6, **cp)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
print(f"  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb)}

# ---------------------------------------------------------------
# Segment definitions (computed on the raw, pre-feature dataframe columns
# still present in X_cal/X_val since asof_pitcher_n and li pass through
# unchanged)
# ---------------------------------------------------------------
COLD_THRESH = 50  # asof_pitcher_n < 50 => "cold-start"

def segments_for(X):
    cold = (X["asof_pitcher_n"] < COLD_THRESH).values
    li = X["li"].values
    li_terciles = np.nanquantile(li, [1/3, 2/3])
    li_bucket = np.digitize(li, li_terciles)  # 0=low, 1=mid, 2=high
    return cold, li_bucket


for name, (model, Xc, Xv) in models.items():
    print(f"\n{'='*70}\n{name}\n{'='*70}")
    p_cal = model.predict_proba(Xc)[:, 1]
    p_val = model.predict_proba(Xv)[:, 1]

    # Global calibrator (current approach)
    iso_global = IsotonicRegression(out_of_bounds="clip")
    iso_global.fit(p_cal, y_cal)
    cal_val_global = iso_global.predict(p_val)
    bss_global, _, _ = brier_skill_score(y_val, cal_val_global, p_clim)
    print(f"Global calibrator, overall VAL BSS: {bss_global:.5f}")

    cold_cal, li_cal = segments_for(Xc)
    cold_val, li_val = segments_for(Xv)

    # --- cold-start vs warm segmented calibration ---
    cal_val_segmented = np.empty_like(cal_val_global)
    for seg_mask_cal, seg_mask_val, seg_label in [(cold_cal, cold_val, "cold"), (~cold_cal, ~cold_val, "warm")]:
        if seg_mask_cal.sum() < 500 or seg_mask_val.sum() < 100:
            cal_val_segmented[seg_mask_val] = cal_val_global[seg_mask_val]
            continue
        iso_seg = IsotonicRegression(out_of_bounds="clip")
        iso_seg.fit(p_cal[seg_mask_cal], y_cal[seg_mask_cal])
        cal_val_segmented[seg_mask_val] = iso_seg.predict(p_val[seg_mask_val])
        seg_bss, _, _ = brier_skill_score(y_val[seg_mask_val], cal_val_segmented[seg_mask_val], p_clim)
        global_seg_bss, _, _ = brier_skill_score(y_val[seg_mask_val], cal_val_global[seg_mask_val], p_clim)
        print(f"  [{seg_label:5s}] n_cal={seg_mask_cal.sum():7d} n_val={seg_mask_val.sum():7d}  "
              f"segment-calibrated BSS={seg_bss:.5f}  global-calibrated BSS={global_seg_bss:.5f}")
    bss_coldwarm, _, _ = brier_skill_score(y_val, cal_val_segmented, p_clim)
    print(f"cold/warm-segmented calibrator, overall VAL BSS: {bss_coldwarm:.5f}  (delta vs global: {bss_coldwarm-bss_global:+.5f})")

    # --- li-bucket segmented calibration ---
    cal_val_li = np.empty_like(cal_val_global)
    for b in (0, 1, 2):
        seg_mask_cal, seg_mask_val = li_cal == b, li_val == b
        if seg_mask_cal.sum() < 500 or seg_mask_val.sum() < 100:
            cal_val_li[seg_mask_val] = cal_val_global[seg_mask_val]
            continue
        iso_seg = IsotonicRegression(out_of_bounds="clip")
        iso_seg.fit(p_cal[seg_mask_cal], y_cal[seg_mask_cal])
        cal_val_li[seg_mask_val] = iso_seg.predict(p_val[seg_mask_val])
        seg_bss, _, _ = brier_skill_score(y_val[seg_mask_val], cal_val_li[seg_mask_val], p_clim)
        global_seg_bss, _, _ = brier_skill_score(y_val[seg_mask_val], cal_val_global[seg_mask_val], p_clim)
        print(f"  [li={b}] n_cal={seg_mask_cal.sum():7d} n_val={seg_mask_val.sum():7d}  "
              f"segment-calibrated BSS={seg_bss:.5f}  global-calibrated BSS={global_seg_bss:.5f}")
    bss_li, _, _ = brier_skill_score(y_val, cal_val_li, p_clim)
    print(f"li-bucket-segmented calibrator, overall VAL BSS: {bss_li:.5f}  (delta vs global: {bss_li-bss_global:+.5f})")

print("\nDONE")
