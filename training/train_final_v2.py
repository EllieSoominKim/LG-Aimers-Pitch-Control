"""Train the v2 final production models: tuned LightGBM + tuned CatBoost +
existing HGB, matchup feature added (shrinkage features evaluated and
dropped -- see eval_result_*.json), logistic-stack blending.

TR_final  = seasons 2019-2023, weighted by decay^(2024-season) (per-model
            decay from Optuna tuning; HGB keeps its existing decay=0.005)
CAL_final = season 2024 (isotonic calibration fit + logistic stacker fit)

Matchup feature: expanding asof stats computed over the FULL train.csv in
row order (verified valid chronological proxy) for TR_final/CAL_final;
the static (pitcher_id,batter_id)->(n,rate) lookup table built from ALL of
train.csv is saved separately for script.py to use at inference time on
test.csv (season 2025, entirely after all of train.csv).
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
from sklearn.metrics import log_loss
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
MODEL_DIR = "C:/LG-Aimers-Pitch-Control/model"
os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading train.csv ...")
t0 = time.time()
raw = pd.read_csv(DATA)
print(f"  loaded {raw.shape} in {time.time()-t0:.1f}s")

cats = feat.fit_category_universe(raw)

# Matchup feature: expanding stats over the FULL row-ordered train.csv.
raw = feat.add_matchup_columns_training(raw, k=30.0)
extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL]


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


def load_params(path, default):
    try:
        with open(path) as f:
            p = json.load(f)
        for k_ in ("val_bss_raw", "val_bss_cal", "cal_bss_objective"):
            p.pop(k_, None)
        return p
    except FileNotFoundError:
        return default


lgbm_params = load_params(
    "C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params.json",
    dict(decay=0.005, num_leaves=15, min_child_samples=2000, learning_rate=0.02,
         reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7))
cb_params = load_params(
    "C:/LG-Aimers-Pitch-Control/training/best_catboost_params.json",
    dict(decay=0.005, depth=3, l2_leaf_reg=20.0, min_data_in_leaf=3000, learning_rate=0.02))

# ---------------------------------------------------------------
# LightGBM (tuned)
# ---------------------------------------------------------------
print("\n=== Training LightGBM (tuned) ===")
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
# CatBoost (tuned)
# ---------------------------------------------------------------
print("\n=== Training CatBoost (tuned) ===")
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
# sklearn HGB (existing config, not Optuna-tuned this round)
# ---------------------------------------------------------------
print("\n=== Training sklearn HGB ===")
t0 = time.time()
DECAY_HGB = 0.005
w_tr_hgb = DECAY_HGB ** (2024 - season_tr)
X_tr_hgb, X_cal_hgb = hgb_frame(X_tr), hgb_frame(X_cal)
hgb_ws = HistGradientBoostingClassifier(
    max_iter=0, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=2000,
    l2_regularization=5.0, categorical_features="from_dtype", early_stopping=False,
    warm_start=True, random_state=0)
BLOCK, PATIENCE, CAP = 25, 8, 2000
best_loss, best_iter, no_improve, total = np.inf, 0, 0, 0
for total in range(BLOCK, CAP + 1, BLOCK):
    hgb_ws.max_iter = total
    hgb_ws.fit(X_tr_hgb, y_tr, sample_weight=w_tr_hgb)
    loss = log_loss(y_cal, hgb_ws.predict_proba(X_cal_hgb)[:, 1])
    if loss < best_loss - 1e-6:
        best_loss, best_iter, no_improve = loss, total, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            break
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=2000,
    l2_regularization=5.0, categorical_features="from_dtype", early_stopping=False, random_state=0)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr_hgb)
print(f"  best_iter={best_iter}  ({time.time()-t0:.1f}s)")
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
cal_cal = {}  # calibrated predictions ON CAL itself, to fit the stacker
for name, p in raw_cal.items():
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p, y_cal)
    calibrators[name] = iso
    cal_cal[name] = iso.predict(p)
    np.savez(os.path.join(MODEL_DIR, f"calibration_{name}.npz"),
              x=iso.X_thresholds_, y=iso.y_thresholds_)

# ---------------------------------------------------------------
# Logistic stacker, fit on CAL's own calibrated predictions (same
# single-fit-then-apply pragmatism already used for calibration itself in
# this production script; rigor lives in the cross-fit evaluation scripts).
# ---------------------------------------------------------------
names = ["lightgbm", "catboost", "hgb"]
stack_X_cal = np.column_stack([cal_cal[n] for n in names])
stacker = LogisticRegression()
stacker.fit(stack_X_cal, y_cal)
print(f"\nStacker coefficients: {dict(zip(names, stacker.coef_[0]))}  intercept: {stacker.intercept_[0]}")

with open(os.path.join(MODEL_DIR, "stacker.json"), "w") as f:
    json.dump({"names": names, "coef": stacker.coef_[0].tolist(),
               "intercept": float(stacker.intercept_[0])}, f, indent=2)

# ---------------------------------------------------------------
# Sanity check on CAL_final itself (in-sample-ish; the trustworthy
# out-of-sample number is 0.01385 from evaluate_pipeline.py's
# matchup_only stage on the untouched 2024 VAL fold using the *previous*
# year's split).
# ---------------------------------------------------------------
stack_cal_pred = stacker.predict_proba(stack_X_cal)[:, 1]
p_clim = y_tr.mean()
bss, bs, _ = brier_skill_score(y_cal, stack_cal_pred, p_clim)
print(f"\n[sanity check, in-sample-ish] stacker on CAL_final: Brier={bs:.5f}  BSS={bss:.5f}")

# ---------------------------------------------------------------
# Matchup lookup table for inference + category universe
# ---------------------------------------------------------------
matchup_table = feat.fit_matchup_lookup(raw, k=30.0)
with open(os.path.join(MODEL_DIR, "matchup_lookup.json"), "w") as f:
    json.dump(matchup_table, f)
print(f"\nMatchup lookup table: {len(matchup_table['pairs'])} pairs")

with open(os.path.join(MODEL_DIR, "categories.json"), "w") as f:
    json.dump(cats, f)

print("\nSaved artifacts to", MODEL_DIR)
print(sorted(os.listdir(MODEL_DIR)))
print("\nDONE")
