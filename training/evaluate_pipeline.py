"""Full pipeline evaluation at a given feature-set stage: trains tuned
LightGBM + tuned CatBoost + existing-config HGB, cross-fit calibrates each,
and compares simple-average / logistic-stack / CAL-optimized-weight blends
-- all on the untouched VAL (season 2024) fold.

Usage: python evaluate_pipeline.py <stage>
  stage in {"baseline", "shrinkage", "shrinkage_matchup"}

Feature-set construction happens ONCE up front (matchup columns need the
full train.csv in row order), then TR/CAL/VAL are sliced from it exactly
as in tuning_common.py, so results are directly comparable across stages.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from metrics import brier_skill_score

STAGE = sys.argv[1] if len(sys.argv) > 1 else "baseline"
assert STAGE in ("baseline", "shrinkage", "shrinkage_matchup", "matchup_only")
print(f"=== STAGE: {STAGE} ===")

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
print("Loading train.csv ...")
raw = pd.read_csv(DATA)
season = raw["season"].values
tr_mask = season <= 2022
cal_mask = season == 2023
val_mask = season == 2024

cats = feat.fit_category_universe(raw)

# ---------------------------------------------------------------
# Build the feature matrix for the requested stage
# ---------------------------------------------------------------
extra_numeric_cols = []
raw_for_features = raw

if STAGE in ("shrinkage", "shrinkage_matchup"):
    # Fit shrinkage priors on TR only (keeps the prior itself leak-safe /
    # consistent with p_clim being a TR-only quantity elsewhere).
    priors = feat.fit_shrinkage_priors(raw[tr_mask])
    raw_for_features = feat.add_shrinkage_columns(raw_for_features, priors, k=50.0)
    extra_numeric_cols += feat.SHRUNK_COLS

if STAGE in ("shrinkage_matchup", "matchup_only"):
    # Expanding matchup stats computed ONCE over the full, row-ordered
    # train.csv (verified valid chronological proxy) -- CAL/VAL rows
    # legitimately see matchup history accumulated during TR (and, for
    # VAL, during TR+CAL), exactly like the official asof_* columns.
    raw_for_features = feat.add_matchup_columns_training(raw_for_features, k=30.0)
    extra_numeric_cols += [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL]


def build_X(df):
    X = feat.build_features(df, categories=cats)
    for col in extra_numeric_cols:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    return X


X_all = build_X(raw_for_features)
y_all = raw["control_success"].values.astype(np.float64)

X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
p_clim = y_tr.mean()
print(f"TR={X_tr.shape}  CAL={X_cal.shape}  VAL={X_val.shape}  (extra cols: {extra_numeric_cols})")

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
print(f"  best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")

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
print(f"  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

print("\n=== Training sklearn HGB ===")
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

avg_val = blend_val_X.mean(axis=1)

stacker = LogisticRegression()
stacker.fit(blend_cal_X, y_cal)
stack_val = stacker.predict_proba(blend_val_X)[:, 1]


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

winner = max(rows, key=lambda r: r[2])
print(f"\nWINNER ({STAGE}): {winner[0]}  BSS={winner[2]:.5f}")

out = {"stage": STAGE, "rows": [{"label": l, "bs": bs, "bss": bss} for l, bs, bss in rows],
       "winner": winner[0], "winner_bss": winner[2]}
with open(f"C:/LG-Aimers-Pitch-Control/training/eval_result_{STAGE}.json", "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved eval_result_{STAGE}.json")
print("DONE")
