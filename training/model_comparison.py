"""Compare LightGBM / CatBoost / sklearn HGB (+ calibration) and blends.

Split:
  TR  = seasons 2019-2022, sample-weighted by recency decay
  CAL = season 2023        (early-stopping monitor + calibration fit,
                             cross-fitted internally to avoid leaking into
                             the blend-weight fit)
  VAL = season 2024        (fully held out — the only number we report as
                             "the" comparison metric)
"""

import sys
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score, brier_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
# Chosen from decay_final_sweep.py: BSS improves monotonically as decay
# drops (more weight on recent seasons), fixed climatology reference,
# best at 0.005 (BSS=0.00814 vs 0.00474 at 0.2, 0.00115 unweighted).
DECAY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.005

print(f"Using recency decay = {DECAY}")
print("Loading train.csv ...")
t0 = time.time()
raw = pd.read_csv(DATA)
print(f"  loaded {raw.shape} in {time.time()-t0:.1f}s")

cats = fit_category_universe(raw)
X_all = build_features(raw, categories=cats)
y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values

tr_mask = season <= 2022
cal_mask = season == 2023
val_mask = season == 2024

X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
w_tr = DECAY ** (2024 - season_tr)

print(f"TR={X_tr.shape}  CAL={X_cal.shape}  VAL={X_val.shape}")
p_clim = y_tr.mean()
print(f"Climatology (TR base rate) = {p_clim:.4f}")

# CatBoost needs plain string/py-native categoricals (no pandas Categorical
# dtype, no NaN in cat cols).
def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X

# sklearn HGB's native categorical splits cap cardinality at 255;
# pitcher_id (792) / batter_id (830) exceed that. Drop them from HGB's view
# specifically -- team_id/hand/asof_* already carry the generalizable
# identity signal (see the ID-coverage-gap finding from EDA), so this is a
# deliberate, documented trade-off rather than a workaround.
HGB_DROP_COLS = ["pitcher_id", "batter_id"]
HGB_CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c not in HGB_DROP_COLS]

def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)

# ---------------------------------------------------------------
# Train base learners on TR (weighted), early-stop on CAL
# ---------------------------------------------------------------
# Heavy regularization across the board: lgbm_regularization_probe.py showed
# this weak-signal, near-50/50 problem overfits within a handful of trees
# under "normal" tree-model defaults (num_leaves=63 etc.) -- shallow trees +
# large min-leaf-samples + strong L2 let boosting extract meaningfully more
# signal (BSS 0.0043->0.0054 at fixed decay) before the eval metric turns.
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

print("\n=== Training CatBoost ===")
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
print(f"  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

print("\n=== Training sklearn HistGradientBoostingClassifier ===")
# sklearn's built-in early_stopping uses an internal random split of X_tr,
# which does NOT respect our recency sample_weight (old, near-zero-weight
# rows can dominate that internal validation fold) -- gave a badly
# miscalibrated raw model that ran the full 5000-iter budget (~5 min)
# without ever triggering a stop. Drive early stopping off CAL manually via
# warm_start instead, matching LightGBM/CatBoost's setup.
t0 = time.time()
X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)
hgb = HistGradientBoostingClassifier(
    max_iter=0, learning_rate=0.02, max_leaf_nodes=15,
    min_samples_leaf=2000, l2_regularization=5.0,
    categorical_features="from_dtype", early_stopping=False,
    warm_start=True, random_state=0,
)
BLOCK, PATIENCE, CAP = 25, 8, 2000
best_loss, best_iter, no_improve = np.inf, 0, 0
from sklearn.metrics import log_loss
for total in range(BLOCK, CAP + 1, BLOCK):
    hgb.max_iter = total
    hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
    p_cal_hgb = hgb.predict_proba(X_cal_hgb)[:, 1]
    loss = log_loss(y_cal, p_cal_hgb)
    if loss < best_loss - 1e-6:
        best_loss, best_iter, no_improve = loss, total, 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            break
# refit exactly to the best iteration count found -- fresh (non-warm-start)
# instance, since warm_start can only grow forward, not rewind.
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter, learning_rate=0.02, max_leaf_nodes=15,
    min_samples_leaf=2000, l2_regularization=5.0,
    categorical_features="from_dtype", early_stopping=False, random_state=0,
)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
print(f"  best_iter={best_iter} (searched up to {total})  ({time.time()-t0:.1f}s)")

models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}

# ---------------------------------------------------------------
# Raw predictions
# ---------------------------------------------------------------
raw_cal = {name: m.predict_proba(Xc)[:, 1] for name, (m, Xc, Xv) in models.items()}
raw_val = {name: m.predict_proba(Xv)[:, 1] for name, (m, Xc, Xv) in models.items()}

# ---------------------------------------------------------------
# Isotonic calibration: cross-fitted within CAL for an honest OOF signal
# used later to fit blend weights; a calibrator refit on all of CAL is used
# as the "production" mapping applied to VAL.
# ---------------------------------------------------------------
rng = np.random.RandomState(0)
fold = rng.randint(0, 2, size=len(y_cal))

oof_cal = {}
cal_val = {}
for name, p_cal in raw_cal.items():
    oof = np.empty_like(p_cal)
    for f in (0, 1):
        fit_mask, pred_mask = fold != f, fold == f
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal[fit_mask], y_cal[fit_mask])
        oof[pred_mask] = iso.predict(p_cal[pred_mask])
    oof_cal[name] = oof

    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_full.fit(p_cal, y_cal)
    cal_val[name] = iso_full.predict(raw_val[name])

# ---------------------------------------------------------------
# Blends: simple average + logistic stacking (fit on OOF-calibrated CAL)
# ---------------------------------------------------------------
names = list(models.keys())
blend_cal_X = np.column_stack([oof_cal[n] for n in names])
blend_val_X = np.column_stack([cal_val[n] for n in names])

avg_val = blend_val_X.mean(axis=1)

stacker = LogisticRegression()
stacker.fit(blend_cal_X, y_cal)
stack_val = stacker.predict_proba(blend_val_X)[:, 1]

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'model':30s} {'BS':>10s} {'BSS':>10s}")
rows = []
for name in names:
    bss_raw, bs_raw, _ = brier_skill_score(y_val, raw_val[name], p_clim)
    bss_cal, bs_cal, _ = brier_skill_score(y_val, cal_val[name], p_clim)
    rows.append((f"{name} (raw)", bs_raw, bss_raw))
    rows.append((f"{name} (calibrated)", bs_cal, bss_cal))

bss_avg, bs_avg, _ = brier_skill_score(y_val, avg_val, p_clim)
bss_stack, bs_stack, _ = brier_skill_score(y_val, stack_val, p_clim)
rows.append(("blend: simple average", bs_avg, bss_avg))
rows.append(("blend: logistic stack", bs_stack, bss_stack))

for label, bs, bss in sorted(rows, key=lambda r: -r[2]):
    print(f"{label:30s} {bs:10.5f} {bss:10.5f}")

print("\nStacker coefficients:", dict(zip(names, stacker.coef_[0])), "intercept:", stacker.intercept_[0])
print("\nDONE")
