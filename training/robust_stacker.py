"""Round 3, item #6: more robust stacker fitting.

Interpretation note: a literal "5-fold over TR+CAL" would require
refitting the base learners on rotating TR+CAL folds too (proper nested
stacked generalization), which is ~5x the already-expensive base-model
training cost (CatBoost especially) for a 3-input logistic regression that
plausibly doesn't need that much data to fit stably. Implemented instead:
5-fold cross-fitting WITHIN CAL (up from the production pipeline's
single-fit-on-all-of-CAL pattern) -- more folds than the round-2
comparison scripts' 2-fold, giving a more stable out-of-fold calibration +
stacker signal without the 5x base-model retraining cost. Also explicitly
tunes the stacker's L2 strength (C) via the same cross-fit OOF BSS, rather
than using LogisticRegression()'s default C=1.0 unexamined.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

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


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


def load_params(paths, default):
    for path in paths:
        try:
            with open(path) as f:
                p = json.load(f)
            for k_ in list(p.keys()):
                # NOTE: "best_iteration" is intentionally NOT stripped here --
                # HGB needs it (fixed max_iter, no early_stopping at inference
                # time), popped explicitly below. Stripping it silently
                # defaulted to 250 instead of the tuned value -- caught before
                # this ran to completion.
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "note"):
                    p.pop(k_, None)
            print(f"Loaded params from {path}")
            return p
        except FileNotFoundError:
            continue
    print(f"None of {paths} found, using default")
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
hgb_params = load_params(
    ["C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json"],
    dict(decay=0.005, learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=2000,
         l2_regularization=5.0, best_iteration=250))

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

print("\n=== Training sklearn HGB ===")
t0 = time.time()
hp = dict(hgb_params)
decay_hgb = hp.pop("decay")
best_iter_hgb = hp.pop("best_iteration", 250)
w_tr_hgb = decay_hgb ** (2024 - season_tr)
X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter_hgb, categorical_features="from_dtype", early_stopping=False,
    random_state=0, **hp)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr_hgb)
print(f"  max_iter={best_iter_hgb}  ({time.time()-t0:.1f}s)")

models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}
names = list(models.keys())

raw_cal = {name: m.predict_proba(Xc)[:, 1] for name, (m, Xc, Xv) in models.items()}
raw_val = {name: m.predict_proba(Xv)[:, 1] for name, (m, Xc, Xv) in models.items()}

# ---------------------------------------------------------------
# 5-fold cross-fit calibration on CAL (up from 2-fold)
# ---------------------------------------------------------------
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=0)
fold_id = np.empty(len(y_cal), dtype=int)
for i, (_, test_idx) in enumerate(kf.split(y_cal)):
    fold_id[test_idx] = i

oof_cal, cal_val = {}, {}
for name, p_cal in raw_cal.items():
    oof = np.empty_like(p_cal)
    for f in range(N_FOLDS):
        fit_mask, pred_mask = fold_id != f, fold_id == f
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal[fit_mask], y_cal[fit_mask])
        oof[pred_mask] = iso.predict(p_cal[pred_mask])
    oof_cal[name] = oof
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(p_cal, y_cal)
    cal_val[name] = iso_full.predict(raw_val[name])
    bss, _, _ = brier_skill_score(y_val, cal_val[name], p_clim)
    print(f"{name} (calibrated): VAL BSS = {bss:.5f}")

blend_cal_X = np.column_stack([oof_cal[n] for n in names])
blend_val_X = np.column_stack([cal_val[n] for n in names])

# ---------------------------------------------------------------
# Stacker C sweep, evaluated via 5-fold cross-fit OOF BSS (not touching VAL)
# ---------------------------------------------------------------
print("\n=== Stacker regularization (C) sweep, 5-fold cross-fit on CAL OOF ===")
best_C, best_cv_bss = None, -np.inf
for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    oof_stack = np.empty(len(y_cal))
    for f in range(N_FOLDS):
        fit_mask, pred_mask = fold_id != f, fold_id == f
        stacker_f = LogisticRegression(C=C)
        stacker_f.fit(blend_cal_X[fit_mask], y_cal[fit_mask])
        oof_stack[pred_mask] = stacker_f.predict_proba(blend_cal_X[pred_mask])[:, 1]
    cv_bss, _, _ = brier_skill_score(y_cal, oof_stack, p_clim)
    print(f"  C={C:<8g}  5-fold OOF CAL BSS = {cv_bss:.5f}")
    if cv_bss > best_cv_bss:
        best_cv_bss, best_C = cv_bss, C
print(f"Best C = {best_C}  (OOF CAL BSS = {best_cv_bss:.5f})")

# Fit final stacker on all of CAL's OOF-calibrated predictions with the
# chosen C, evaluate on the untouched VAL fold.
stacker = LogisticRegression(C=best_C)
stacker.fit(blend_cal_X, y_cal)
stack_val = stacker.predict_proba(blend_val_X)[:, 1]
bss_stack, bs_stack, _ = brier_skill_score(y_val, stack_val, p_clim)
print(f"\nRobust (5-fold, tuned C={best_C}) stacker: VAL BSS = {bss_stack:.5f}")

# Compare against the round-2-style single C=1.0, 2-fold stacker for a
# clean delta.
kf2 = KFold(n_splits=2, shuffle=True, random_state=0)
fold2_id = np.empty(len(y_cal), dtype=int)
for i, (_, test_idx) in enumerate(kf2.split(y_cal)):
    fold2_id[test_idx] = i
oof_cal_2fold = {}
for name, p_cal in raw_cal.items():
    oof = np.empty_like(p_cal)
    for f in range(2):
        fit_mask, pred_mask = fold2_id != f, fold2_id == f
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal[fit_mask], y_cal[fit_mask])
        oof[pred_mask] = iso.predict(p_cal[pred_mask])
    oof_cal_2fold[name] = oof
blend_cal_X_2fold = np.column_stack([oof_cal_2fold[n] for n in names])
stacker_baseline = LogisticRegression(C=1.0)
stacker_baseline.fit(blend_cal_X_2fold, y_cal)
stack_val_baseline = stacker_baseline.predict_proba(blend_val_X)[:, 1]
bss_baseline, _, _ = brier_skill_score(y_val, stack_val_baseline, p_clim)
print(f"Baseline (2-fold, C=1.0) stacker for comparison: VAL BSS = {bss_baseline:.5f}")
print(f"Delta: {bss_stack - bss_baseline:+.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/robust_stacker_result.json", "w") as f:
    json.dump({"names": names, "coef": stacker.coef_[0].tolist(),
               "intercept": float(stacker.intercept_[0]), "C": best_C,
               "val_bss_robust": bss_stack, "val_bss_baseline_2fold": bss_baseline,
               "delta": bss_stack - bss_baseline}, f, indent=2)
print("\nDONE")
