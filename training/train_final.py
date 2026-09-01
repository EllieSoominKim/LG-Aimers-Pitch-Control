"""Train the final production models on the full dataset and save artifacts
to model/, ready for script.py to load.

Shifts the validated scheme forward by one year relative to
model_comparison.py:
  TR_final  = seasons 2019-2023, weighted by decay^(2024-season)
  CAL_final = season 2024 (most recent complete season -> isotonic
              calibration fit for each base model)

Final prediction = simple average of the 3 calibrated model outputs
(the blend that won the held-out 2024 comparison in model_comparison.py).
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
from sklearn.metrics import log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
MODEL_DIR = "C:/LG-Aimers-Pitch-Control/model"
DECAY = 0.005
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading train.csv ...")
t0 = time.time()
raw = pd.read_csv(DATA)
print(f"  loaded {raw.shape} in {time.time()-t0:.1f}s")

cats = fit_category_universe(raw)
X_all = build_features(raw, categories=cats)
y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values

tr_mask = season <= 2023
cal_mask = season == 2024

X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
w_tr = DECAY ** (2024 - season_tr)
print(f"TR_final={X_tr.shape}  CAL_final={X_cal.shape}")


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


# ---------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------
print("\n=== Training LightGBM ===")
t0 = time.time()
lgbm = lgb.LGBMClassifier(
    n_estimators=5000, learning_rate=0.02, num_leaves=15,
    min_child_samples=2000, reg_lambda=5.0, subsample=0.7,
    colsample_bytree=0.7, objective="binary", verbosity=-1, n_jobs=6,
)
lgbm.fit(
    X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
    eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(100, verbose=False)],
)
print(f"  best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")
lgbm.booster_.save_model(os.path.join(MODEL_DIR, "lightgbm.txt"))

# ---------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------
print("\n=== Training CatBoost ===")
t0 = time.time()
X_tr_cb, X_cal_cb = catboost_frame(X_tr), catboost_frame(X_cal)
cb = CatBoostClassifier(
    iterations=5000, learning_rate=0.02, depth=3,
    l2_leaf_reg=20.0, min_data_in_leaf=3000,
    loss_function="Logloss", eval_metric="Logloss",
    cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=150,
    thread_count=6,
)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
print(f"  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")
cb.save_model(os.path.join(MODEL_DIR, "catboost.cbm"))

# ---------------------------------------------------------------
# sklearn HGB (manual warm-start early stopping against CAL)
# ---------------------------------------------------------------
print("\n=== Training sklearn HistGradientBoostingClassifier ===")
t0 = time.time()
X_tr_hgb, X_cal_hgb = hgb_frame(X_tr), hgb_frame(X_cal)
hgb_ws = HistGradientBoostingClassifier(
    max_iter=0, learning_rate=0.02, max_leaf_nodes=15,
    min_samples_leaf=2000, l2_regularization=5.0,
    categorical_features="from_dtype", early_stopping=False,
    warm_start=True, random_state=0,
)
BLOCK, PATIENCE, CAP = 25, 8, 2000
best_loss, best_iter, no_improve, total = np.inf, 0, 0, 0
for total in range(BLOCK, CAP + 1, BLOCK):
    hgb_ws.max_iter = total
    hgb_ws.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
    p_cal_hgb = hgb_ws.predict_proba(X_cal_hgb)[:, 1]
    loss = log_loss(y_cal, p_cal_hgb)
    if loss < best_loss - 1e-6:
        best_loss, best_iter, no_improve = loss, total, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            break
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter, learning_rate=0.02, max_leaf_nodes=15,
    min_samples_leaf=2000, l2_regularization=5.0,
    categorical_features="from_dtype", early_stopping=False, random_state=0,
)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
print(f"  best_iter={best_iter} (searched up to {total})  ({time.time()-t0:.1f}s)")
joblib.dump(hgb, os.path.join(MODEL_DIR, "hgb.joblib"))

# ---------------------------------------------------------------
# Isotonic calibration per model, fit on CAL_final (season 2024)
# ---------------------------------------------------------------
print("\n=== Fitting isotonic calibration (on season-2024 CAL fold) ===")
raw_cal = {
    "lightgbm": lgbm.predict_proba(X_cal)[:, 1],
    "catboost": cb.predict_proba(X_cal_cb)[:, 1],
    "hgb": hgb.predict_proba(X_cal_hgb)[:, 1],
}
calibrators = {}
for name, p in raw_cal.items():
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y_cal)
    calibrators[name] = iso
    # Save as plain (x, y) breakpoint arrays -- portable, no sklearn object
    # needed at inference time, just np.interp.
    np.savez(
        os.path.join(MODEL_DIR, f"calibration_{name}.npz"),
        x=iso.X_thresholds_, y=iso.y_thresholds_,
    )

# ---------------------------------------------------------------
# Sanity check: calibrated blend BSS on CAL_final itself (in-sample-ish;
# the trustworthy out-of-sample number is the 0.01270 found in
# model_comparison.py on the untouched 2024 fold using the *previous*
# year's split -- this print is just a sanity check the pipeline reproduces
# comparable behavior, not a fresh unbiased estimate).
# ---------------------------------------------------------------
cal_preds = np.column_stack([calibrators[n].predict(raw_cal[n]) for n in raw_cal])
blend_cal = cal_preds.mean(axis=1)
p_clim = y_tr.mean()
bss, bs, _ = brier_skill_score(y_cal, blend_cal, p_clim)
print(f"\n[sanity check, in-sample-ish] blend on CAL_final: Brier={bs:.5f}  BSS={bss:.5f}")

# ---------------------------------------------------------------
# Save category universe + feature column metadata for script.py
# ---------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "categories.json"), "w") as f:
    json.dump(cats, f)

print("\nSaved artifacts to", MODEL_DIR)
print(os.listdir(MODEL_DIR))
print("\nDONE")
