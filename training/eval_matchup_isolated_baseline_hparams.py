"""Isolate the base matchup feature's effect under the EXACT 622.14
baseline's hyperparameters/blend -- something never actually measured
before. Every prior matchup evaluation in this project used Optuna-tuned
hyperparameters, which is a confound: it could not tell us whether matchup
itself helps, independent of also having re-tuned everything else.

Trains LightGBM/CatBoost/HGB with train_final.py's exact fixed manual
hyperparameters and DECAY=0.005 (uniform across all 3 models, matching
what's actually being deployed), on TR=2019-2022 / CAL=2023 / VAL=2024
(model_comparison.py's protocol, one year earlier than production's
TR_final/CAL_final so VAL=2024 stays genuinely held out), comparing:
  (a) no matchup feature at all (53 features, exact 622.14 config)
  (b) + base matchup feature only (54 features, leak-fixed
      add_matchup_columns_training -- see verify_extension_leakage.py)
Blend: simple average of 3 globally-calibrated models (matching
train_final.py, NOT the logistic stacker used in later rounds).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
DECAY = 0.005  # train_final.py's fixed, uniform decay -- NOT the per-model Optuna-tuned values

print("Loading train.csv ...")
raw = pd.read_csv(DATA)
cats = fit_category_universe(raw)
season = raw["season"].values
tr_mask = season <= 2022
cal_mask = season == 2023
val_mask = season == 2024


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


def run(include_matchup):
    label = "622.14 baseline + matchup (leak-fixed)" if include_matchup else "622.14 baseline (no matchup)"
    print(f"\n{'='*70}\n{label}\n{'='*70}")

    raw_for_features = raw
    extra_cols = []
    if include_matchup:
        raw_for_features = feat.add_matchup_columns_training(raw_for_features, k=30.0)
        extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL]

    X_all = build_features(raw_for_features, categories=cats)
    for col in extra_cols:
        X_all[col] = pd.to_numeric(raw_for_features[col], errors="coerce").astype(np.float64)

    y_all = raw["control_success"].values.astype(np.float64)
    X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    p_clim = y_tr.mean()
    w_tr = DECAY ** (2024 - season_tr)
    print(f"TR={X_tr.shape}  CAL={X_cal.shape}  VAL={X_val.shape}  features={list(X_all.columns)[-3:]}")

    # --- LightGBM (train_final.py's exact hyperparameters) ---
    t0 = time.time()
    lgbm = lgb.LGBMClassifier(
        n_estimators=5000, learning_rate=0.02, num_leaves=15,
        min_child_samples=2000, reg_lambda=5.0, subsample=0.7,
        colsample_bytree=0.7, objective="binary", verbosity=-1, n_jobs=6,
    )
    lgbm.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
             eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
             callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  lightgbm best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")

    # --- CatBoost (train_final.py's exact hyperparameters) ---
    t0 = time.time()
    X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
    cb = CatBoostClassifier(
        iterations=5000, learning_rate=0.02, depth=3,
        l2_leaf_reg=20.0, min_data_in_leaf=3000,
        loss_function="Logloss", eval_metric="Logloss",
        cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=150,
        thread_count=6,
    )
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
    print(f"  catboost best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

    # --- sklearn HGB (train_final.py's exact hyperparameters + warm-start loop) ---
    t0 = time.time()
    X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)
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
        loss = log_loss(y_cal, hgb_ws.predict_proba(X_cal_hgb)[:, 1])
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
    print(f"  hgb best_iter={best_iter}  ({time.time()-t0:.1f}s)")

    # --- global isotonic calibration + simple average blend (train_final.py's approach) ---
    models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}
    cal_val = {}
    for name, (m, Xc, Xv) in models.items():
        p_cal = m.predict_proba(Xc)[:, 1]
        p_val = m.predict_proba(Xv)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        cal_val[name] = iso.predict(p_val)
        bss, _, _ = brier_skill_score(y_val, cal_val[name], p_clim)
        print(f"  {name} (calibrated): VAL BSS = {bss:.5f}")

    avg_val = np.mean([cal_val[n] for n in ("lightgbm", "catboost", "hgb")], axis=0)
    bss_avg, bs_avg, _ = brier_skill_score(y_val, avg_val, p_clim)
    print(f"  SIMPLE AVERAGE BLEND: VAL BSS = {bss_avg:.5f}  (Brier={bs_avg:.5f})")
    return bss_avg, {n: brier_skill_score(y_val, cal_val[n], p_clim)[0] for n in cal_val}


bss_no_matchup, per_model_no = run(False)
bss_matchup, per_model_yes = run(True)

print(f"\n{'='*70}\nSummary\n{'='*70}")
print(f"622.14 baseline (no matchup):          VAL BSS = {bss_no_matchup:.5f}")
print(f"622.14 baseline + matchup (leak-fixed): VAL BSS = {bss_matchup:.5f}")
print(f"Delta: {bss_matchup - bss_no_matchup:+.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/eval_matchup_isolated_result.json", "w") as f:
    json.dump({
        "no_matchup_val_bss": bss_no_matchup, "no_matchup_per_model": per_model_no,
        "matchup_val_bss": bss_matchup, "matchup_per_model": per_model_yes,
        "delta": bss_matchup - bss_no_matchup,
    }, f, indent=2)
print("\nDONE")
