"""Round 3, item #7: multi-seed ensembling. Train each base model (using
its tuned params) with 2-3 different random seeds, average the calibrated
predictions within each model family before blending across families --
tests whether variance reduction helps given how weak the overall signal
is (BSS ~0.01-0.02 range throughout this project).

LightGBM/CatBoost don't have a single "random_state" that's the sole
source of randomness relevant here (subsample/colsample/bagging already
provide stochasticity), but both accept an explicit seed controlling that
stochasticity -- vary it across runs. HGB varies via random_state directly.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

SEEDS = [0, 1, 2]

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
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "best_iteration"):
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

X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)


def calibrated_val_bss(raw_cal, raw_val):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)
    cal_val = iso.predict(raw_val)
    bss, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    return bss, cal_val


print("=== LightGBM across seeds ===")
lp = dict(lgbm_params)
decay_lgbm = lp.pop("decay")
w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
lgbm_cal_vals = []
for seed in SEEDS:
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6,
                            random_state=seed, bagging_seed=seed, feature_fraction_seed=seed, **lp)
    m.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
          eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False)])
    raw_cal, raw_val = m.predict_proba(X_cal)[:, 1], m.predict_proba(X_val)[:, 1]
    bss, cal_val = calibrated_val_bss(raw_cal, raw_val)
    lgbm_cal_vals.append(cal_val)
    print(f"  seed={seed}  best_iter={m.best_iteration_}  VAL BSS={bss:.5f}  ({time.time()-t0:.1f}s)")
lgbm_avg_val = np.mean(lgbm_cal_vals, axis=0)
bss_lgbm_avg, _, _ = brier_skill_score(y_val, lgbm_avg_val, p_clim)
print(f"  Multi-seed AVERAGE: VAL BSS={bss_lgbm_avg:.5f}")

print("\n=== CatBoost across seeds ===")
cp = dict(cb_params)
decay_cb = cp.pop("decay")
w_tr_cb = decay_cb ** (2024 - season_tr)
cb_cal_vals = []
for seed in SEEDS:
    t0 = time.time()
    m = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                            cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                            thread_count=6, random_seed=seed, **cp)
    m.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
    raw_cal, raw_val = m.predict_proba(X_cal_cb)[:, 1], m.predict_proba(X_val_cb)[:, 1]
    bss, cal_val = calibrated_val_bss(raw_cal, raw_val)
    cb_cal_vals.append(cal_val)
    print(f"  seed={seed}  best_iter={m.get_best_iteration()}  VAL BSS={bss:.5f}  ({time.time()-t0:.1f}s)")
cb_avg_val = np.mean(cb_cal_vals, axis=0)
bss_cb_avg, _, _ = brier_skill_score(y_val, cb_avg_val, p_clim)
print(f"  Multi-seed AVERAGE: VAL BSS={bss_cb_avg:.5f}")

print(f"\n{'='*70}")
print(f"LightGBM: single-seed(0)={brier_skill_score(y_val, lgbm_cal_vals[0], p_clim)[0]:.5f}  "
      f"multi-seed-avg={bss_lgbm_avg:.5f}  delta={bss_lgbm_avg - brier_skill_score(y_val, lgbm_cal_vals[0], p_clim)[0]:+.5f}")
print(f"CatBoost: single-seed(0)={brier_skill_score(y_val, cb_cal_vals[0], p_clim)[0]:.5f}  "
      f"multi-seed-avg={bss_cb_avg:.5f}  delta={bss_cb_avg - brier_skill_score(y_val, cb_cal_vals[0], p_clim)[0]:+.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/multiseed_result.json", "w") as f:
    json.dump({
        "seeds": SEEDS,
        "lightgbm_single_seed0_bss": brier_skill_score(y_val, lgbm_cal_vals[0], p_clim)[0],
        "lightgbm_multiseed_avg_bss": bss_lgbm_avg,
        "catboost_single_seed0_bss": brier_skill_score(y_val, cb_cal_vals[0], p_clim)[0],
        "catboost_multiseed_avg_bss": bss_cb_avg,
    }, f, indent=2)
print("\nDONE")
