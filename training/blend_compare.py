"""Step 4: blend upgrade. Compares simple average vs logistic-regression
stacker vs validation-optimized weighted average, using the (Optuna-tuned,
once available) LightGBM + CatBoost configs plus the existing HGB config.

Blend weights/stacker are fit on CAL's cross-fit OOF-calibrated predictions
(no leakage into VAL); VAL (season 2024) is the untouched final comparison,
exactly as in model_comparison.py.

Usage: python blend_compare.py [path_to_lgbm_params.json] [path_to_catboost_params.json]
       (falls back to the current hardcoded defaults if files are absent)
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

d = load_split()
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


# ---------------------------------------------------------------
# Load tuned params (fall back to current defaults if not found)
# ---------------------------------------------------------------
def load_params(path, default):
    try:
        with open(path) as f:
            p = json.load(f)
        p.pop("val_bss_raw", None)
        p.pop("val_bss_cal", None)
        p.pop("cal_bss_objective", None)
        print(f"Loaded tuned params from {path}")
        return p
    except FileNotFoundError:
        print(f"{path} not found, using default params")
        return default


lgbm_params = load_params(
    "C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params.json",
    dict(decay=0.005, num_leaves=15, min_child_samples=2000, learning_rate=0.02,
         reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7),
)
cb_params = load_params(
    "C:/LG-Aimers-Pitch-Control/training/best_catboost_params.json",
    dict(decay=0.005, depth=3, l2_leaf_reg=20.0, min_data_in_leaf=3000, learning_rate=0.02),
)

# ---------------------------------------------------------------
# Train base learners
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

print("\n=== Training CatBoost (tuned) ===")
t0 = time.time()
cp = dict(cb_params)
decay_cb = cp.pop("decay")
w_tr_cb = decay_cb ** (2024 - season_tr)
X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                         cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                         thread_count=6, **cp)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
print(f"  best_iter={cb.get_best_iteration()}  decay={decay_cb:.4f}  ({time.time()-t0:.1f}s)")

print("\n=== Training sklearn HGB (existing config, not Optuna-tuned this round) ===")
t0 = time.time()
DECAY_HGB = 0.005
w_tr_hgb = DECAY_HGB ** (2024 - season_tr)
X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)
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

models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}

# ---------------------------------------------------------------
# Raw predictions + cross-fitted calibration (same as model_comparison.py)
# ---------------------------------------------------------------
raw_cal = {name: m.predict_proba(Xc)[:, 1] for name, (m, Xc, Xv) in models.items()}
raw_val = {name: m.predict_proba(Xv)[:, 1] for name, (m, Xc, Xv) in models.items()}

rng = np.random.RandomState(0)
fold = rng.randint(0, 2, size=len(y_cal))
oof_cal, cal_val = {}, {}
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

names = list(models.keys())
blend_cal_X = np.column_stack([oof_cal[n] for n in names])
blend_val_X = np.column_stack([cal_val[n] for n in names])

# ---------------------------------------------------------------
# Blend option A: simple average
# ---------------------------------------------------------------
avg_val = blend_val_X.mean(axis=1)

# ---------------------------------------------------------------
# Blend option B: logistic-regression stacker (fit on CAL OOF)
# ---------------------------------------------------------------
stacker = LogisticRegression()
stacker.fit(blend_cal_X, y_cal)
stack_val = stacker.predict_proba(blend_val_X)[:, 1]

# ---------------------------------------------------------------
# Blend option C: validation(CAL)-optimized weighted average (weights sum
# to 1, each >=0), maximizing BSS on CAL's OOF-calibrated predictions.
# ---------------------------------------------------------------
def neg_bss_for_weights(w, X, y, p_clim):
    w = np.abs(w)
    w = w / w.sum()
    p = X @ w
    bss, _, _ = brier_skill_score(y, p, p_clim)
    return -bss

w0 = np.ones(len(names)) / len(names)
res = minimize(neg_bss_for_weights, w0, args=(blend_cal_X, y_cal, p_clim), method="Nelder-Mead")
opt_w = np.abs(res.x)
opt_w = opt_w / opt_w.sum()
optw_val = blend_val_X @ opt_w

# ---------------------------------------------------------------
# Report
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'model':30s} {'BS':>10s} {'BSS':>10s}")
rows = []
for name in names:
    bss_cal, bs_cal, _ = brier_skill_score(y_val, cal_val[name], p_clim)
    rows.append((f"{name} (calibrated)", bs_cal, bss_cal))
bss_avg, bs_avg, _ = brier_skill_score(y_val, avg_val, p_clim)
bss_stack, bs_stack, _ = brier_skill_score(y_val, stack_val, p_clim)
bss_optw, bs_optw, _ = brier_skill_score(y_val, optw_val, p_clim)
rows.append(("blend: simple average", bs_avg, bss_avg))
rows.append(("blend: logistic stack", bs_stack, bss_stack))
rows.append(("blend: CAL-optimized weights", bs_optw, bss_optw))

for label, bs, bss in sorted(rows, key=lambda r: -r[2]):
    print(f"{label:30s} {bs:10.5f} {bss:10.5f}")

print("\nStacker coefficients:", dict(zip(names, stacker.coef_[0])), "intercept:", stacker.intercept_[0])
print("Optimized weights:", dict(zip(names, opt_w)))

winner = max(rows, key=lambda r: r[2])
print(f"\nWINNER: {winner[0]}  BSS={winner[2]:.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/blend_compare_result.json", "w") as f:
    json.dump({"rows": [{"label": l, "bs": bs, "bss": bss} for l, bs, bss in rows],
               "stacker_coef": dict(zip(names, stacker.coef_[0].tolist())),
               "stacker_intercept": float(stacker.intercept_[0]),
               "optimized_weights": dict(zip(names, opt_w.tolist())),
               "winner": winner[0]}, f, indent=2)
print("\nDONE")
