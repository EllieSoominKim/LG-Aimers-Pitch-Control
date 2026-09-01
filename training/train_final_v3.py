"""Train the v3 final production models -- round 4's validated
configuration:
  - matchup feature + extensions (last-N trend, pitcher-vs-hand rate,
    interactions) -- item #4, robust across seeds, +0.00105/+0.00080 avg
  - Optuna-retuned LightGBM/CatBoost (400 trials, matchup-aware) -- items #1+#3
  - Optuna-tuned HGB (stopped early at 221/400 trials after a 41-trial
    plateau; tuned on base-matchup-only features, accepted as-is per
    explicit decision -- item #2/#7 discussion)
  - cold/warm segmented isotonic calibration (asof_pitcher_n<50) -- item #5,
    small but consistent gain, li-bucket segmentation explicitly rejected
  - simple single-fit logistic stacker (5-fold cross-fit + C-tuning showed
    no benefit over this -- item #6)
  - trackman env priors explicitly excluded -- item #8, hurt both models
  - multi-seed ensembling explicitly excluded -- item #7, marginal/mixed,
    not worth tripling the embedded artifact count

TR_final  = seasons 2019-2023, weighted by decay^(2024-season) (per-model
            decay from Optuna tuning; HGB keeps its own tuned decay)
CAL_final = season 2024 (segmented isotonic calibration fit + stacker fit)

The prior_mean look-ahead bug in add_matchup_columns_training /
add_matchup_extension_columns_training (see verify_extension_leakage.py)
was fixed in features.py before this script was written -- both TR_final
and CAL_final's matchup/extension features here are leak-safe.
"""
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
MODEL_DIR = "C:/LG-Aimers-Pitch-Control/model"
os.makedirs(MODEL_DIR, exist_ok=True)

COLD_THRESH = 50  # matches investigate_slice_calibration.py's item #5 threshold

print("Loading train.csv ...")
t0 = time.time()
raw = pd.read_csv(DATA)
print(f"  loaded {raw.shape} in {time.time()-t0:.1f}s")

cats = feat.fit_category_universe(raw)

# Matchup + extension features: expanding stats over the FULL row-ordered
# train.csv (leak-safe as of the features.py fix).
raw = feat.add_matchup_columns_training(raw, k=30.0)
raw = feat.add_matchup_extension_columns_training(raw, k=30.0)
extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL] + feat.MATCHUP_EXTENSION_COLS
print(f"  extra numeric cols (order matters for CatBoost): {extra_cols}")


def build_X(df):
    X = feat.build_features(df, categories=cats)
    for col in extra_cols:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    return X


X_all = build_X(raw)
y_all = raw["control_success"].values.astype(np.float64)
season = raw["season"].values

tr_mask = season <= 2023
cal_mask = season == 2024

X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
print(f"TR_final={X_tr.shape}  CAL_final={X_cal.shape}")

CATEGORICAL_COLS = feat.CATEGORICAL_COLS


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


def load_params(paths, default):
    for path in paths:
        try:
            with open(path) as f:
                p = json.load(f)
            for k_ in list(p.keys()):
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "note", "model_size_mb"):
                    p.pop(k_, None)
            print(f"Loaded params from {path}")
            return p
        except FileNotFoundError:
            continue
    raise FileNotFoundError(f"None of {paths} found -- refusing to fall back to defaults for production training")


lgbm_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json"], None)
cb_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v3.json"], None)
hgb_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json"], None)

# ---------------------------------------------------------------
# LightGBM (v2 tuned, matchup-aware)
# ---------------------------------------------------------------
print("\n=== Training LightGBM (v2 tuned) ===")
t0 = time.time()
lp = dict(lgbm_params)
decay_lgbm = lp.pop("decay")
w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
lgbm = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **lp)
lgbm.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
         eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
         callbacks=[lgb.early_stopping(80, verbose=False)])
print(f"  best_iter={lgbm.best_iteration_}  decay={decay_lgbm:.4f}  ({time.time()-t0:.1f}s)")
lgbm.booster_.save_model(os.path.join(MODEL_DIR, "lightgbm.txt"))

# ---------------------------------------------------------------
# CatBoost (v2 tuned, matchup-aware)
# ---------------------------------------------------------------
print("\n=== Training CatBoost (v2 tuned) ===")
t0 = time.time()
cp = dict(cb_params)
decay_cb = cp.pop("decay")
w_tr_cb = decay_cb ** (2024 - season_tr)
X_tr_cb, X_cal_cb = catboost_frame(X_tr), catboost_frame(X_cal)
cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                         cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                         thread_count=6, **cp)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
print(f"  best_iter={cb.get_best_iteration()}  decay={decay_cb:.4f}  ({time.time()-t0:.1f}s)")
cb.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))

# ---------------------------------------------------------------
# sklearn HGB (Optuna-tuned, item #2 -- accepted as-is per explicit
# decision, though tuned on base-matchup-only features)
# ---------------------------------------------------------------
print("\n=== Training sklearn HGB (tuned) ===")
t0 = time.time()
hp = dict(hgb_params)
decay_hgb = hp.pop("decay")
best_iter_hgb = hp.pop("best_iteration")
w_tr_hgb = decay_hgb ** (2024 - season_tr)
X_tr_hgb, X_cal_hgb = hgb_frame(X_tr), hgb_frame(X_cal)
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter_hgb, categorical_features="from_dtype", early_stopping=False,
    random_state=0, **hp)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr_hgb)
print(f"  max_iter={best_iter_hgb}  decay={decay_hgb:.4f}  ({time.time()-t0:.1f}s)")
joblib.dump(hgb, os.path.join(MODEL_DIR, "hgb.joblib"))

# ---------------------------------------------------------------
# Cold/warm segmented isotonic calibration, fit on CAL_final (season 2024)
# -- item #5. Each model gets a "cold" calibrator (fit only on
# asof_pitcher_n<COLD_THRESH CAL rows) and a "warm" calibrator (the rest).
# ---------------------------------------------------------------
print("\n=== Fitting cold/warm segmented isotonic calibration (CAL_final=2024) ===")
raw_cal = {
    "lightgbm": lgbm.predict_proba(X_cal)[:, 1],
    "catboost": cb.predict_proba(X_cal_cb)[:, 1],
    "hgb": hgb.predict_proba(X_cal_hgb)[:, 1],
}
cold_mask_cal = (X_cal["asof_pitcher_n"] < COLD_THRESH).values
print(f"  CAL_final cold rows: {cold_mask_cal.sum()} / {len(cold_mask_cal)}")

calibrators = {}       # name -> {"cold": IsotonicRegression, "warm": IsotonicRegression}
cal_cal_segmented = {}  # name -> calibrated predictions on CAL_final itself (for stacker fit)
for name, p in raw_cal.items():
    iso_cold = IsotonicRegression(out_of_bounds="clip")
    iso_cold.fit(p[cold_mask_cal], y_cal[cold_mask_cal])
    iso_warm = IsotonicRegression(out_of_bounds="clip")
    iso_warm.fit(p[~cold_mask_cal], y_cal[~cold_mask_cal])
    calibrators[name] = {"cold": iso_cold, "warm": iso_warm}

    seg_pred = np.empty_like(p)
    seg_pred[cold_mask_cal] = iso_cold.predict(p[cold_mask_cal])
    seg_pred[~cold_mask_cal] = iso_warm.predict(p[~cold_mask_cal])
    cal_cal_segmented[name] = seg_pred

    np.savez(os.path.join(MODEL_DIR, f"calibration_{name}.npz"),
             x_cold=iso_cold.X_thresholds_, y_cold=iso_cold.y_thresholds_,
             x_warm=iso_warm.X_thresholds_, y_warm=iso_warm.y_thresholds_)

# ---------------------------------------------------------------
# Logistic stacker, fit on CAL_final's own (segment-calibrated)
# predictions -- single fit, matching the production pragmatism validated
# in item #6 (5-fold cross-fit + C-tuning showed no benefit).
# ---------------------------------------------------------------
names = ["lightgbm", "catboost", "hgb"]
stack_X_cal = np.column_stack([cal_cal_segmented[n] for n in names])
stacker = LogisticRegression()
stacker.fit(stack_X_cal, y_cal)
print(f"\nStacker coefficients: {dict(zip(names, stacker.coef_[0]))}  intercept: {stacker.intercept_[0]}")

with open(os.path.join(MODEL_DIR, "stacker.json"), "w") as f:
    json.dump({"names": names, "coef": stacker.coef_[0].tolist(),
               "intercept": float(stacker.intercept_[0])}, f, indent=2)

# ---------------------------------------------------------------
# Sanity check on CAL_final itself (in-sample-ish; the trustworthy
# out-of-sample number is the round-4 validated 0.0149x-ish figure from
# eval scripts on the untouched 2024 VAL fold using the *previous* year's
# TR/CAL split).
# ---------------------------------------------------------------
stack_cal_pred = stacker.predict_proba(stack_X_cal)[:, 1]
p_clim = y_tr.mean()
bss, bs, _ = brier_skill_score(y_cal, stack_cal_pred, p_clim)
print(f"\n[sanity check, in-sample-ish] stacker on CAL_final: Brier={bs:.5f}  BSS={bss:.5f}")

# ---------------------------------------------------------------
# Matchup + extension lookup tables for inference + category universe
# ---------------------------------------------------------------
matchup_table = feat.fit_matchup_lookup(raw, k=30.0)
with open(os.path.join(MODEL_DIR, "matchup_lookup.json"), "w") as f:
    json.dump(matchup_table, f)
print(f"\nMatchup lookup table: {len(matchup_table['pairs'])} pairs")

ext_table = feat.fit_matchup_extension_lookup(raw)
with open(os.path.join(MODEL_DIR, "matchup_extension_lookup.json"), "w") as f:
    json.dump(ext_table, f)
print(f"Matchup extension lookup table: {len(ext_table['pair_lastn'])} pairs, "
      f"{len(ext_table['pitcher_hand'])} pitcher-hand entries")

with open(os.path.join(MODEL_DIR, "categories.json"), "w") as f:
    json.dump(cats, f)

print("\nSaved artifacts to", MODEL_DIR)
print(sorted(os.listdir(MODEL_DIR)))
print("\nDONE")
