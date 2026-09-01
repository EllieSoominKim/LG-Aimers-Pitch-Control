"""Item #4 multi-seed robustness check: does the base-vs-extensions delta
hold up across different model random seeds, or was the single-seed run
partly luck? Trains LightGBM and CatBoost with seeds [0,1,2,3,4] for both
feature sets, reports the delta's mean and spread.
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

SEEDS = [0, 1, 2, 3, 4]


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
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "note"):
                    p.pop(k_, None)
            print(f"Loaded params from {path}")
            return p
        except FileNotFoundError:
            continue
    print(f"None of {paths} found, using default")
    return default


lgbm_params = load_params(
    ["C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json"], {})
cb_params = load_params(
    ["C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v2.json"], {})


def run(include_extensions):
    d = load_split(include_matchup_extensions=include_extensions)
    X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
    X_cal, y_cal = d["X_cal"], d["y_cal"]
    X_val, y_val = d["X_val"], d["y_val"]
    p_clim = d["p_clim"]

    lgbm_bss, cb_bss = [], []

    lp = dict(lgbm_params)
    decay_lgbm = lp.pop("decay")
    w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
    for seed in SEEDS:
        t0 = time.time()
        m = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6,
                                random_state=seed, bagging_seed=seed, feature_fraction_seed=seed, **lp)
        m.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
              eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
              callbacks=[lgb.early_stopping(80, verbose=False)])
        p_cal, p_val = m.predict_proba(X_cal)[:, 1], m.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        cal_val = iso.predict(p_val)
        bss, _, _ = brier_skill_score(y_val, cal_val, p_clim)
        lgbm_bss.append(bss)
        print(f"    lightgbm seed={seed}  VAL BSS={bss:.5f}  ({time.time()-t0:.1f}s)")

    cp = dict(cb_params)
    decay_cb = cp.pop("decay")
    w_tr_cb = decay_cb ** (2024 - season_tr)
    X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
    for seed in SEEDS:
        t0 = time.time()
        m = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                                cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                                thread_count=6, random_seed=seed, **cp)
        m.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
        p_cal, p_val = m.predict_proba(X_cal_cb)[:, 1], m.predict_proba(X_val_cb)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        cal_val = iso.predict(p_val)
        bss, _, _ = brier_skill_score(y_val, cal_val, p_clim)
        cb_bss.append(bss)
        print(f"    catboost seed={seed}  VAL BSS={bss:.5f}  ({time.time()-t0:.1f}s)")

    return np.array(lgbm_bss), np.array(cb_bss)


print("=== base matchup only ===")
lgbm_base, cb_base = run(False)
print("\n=== matchup + extensions ===")
lgbm_ext, cb_ext = run(True)

lgbm_deltas = lgbm_ext - lgbm_base
cb_deltas = cb_ext - cb_base

print(f"\n{'='*70}\nSummary across {len(SEEDS)} seeds\n{'='*70}")
print(f"lightgbm  base: mean={lgbm_base.mean():.5f} std={lgbm_base.std():.5f}  "
      f"ext: mean={lgbm_ext.mean():.5f} std={lgbm_ext.std():.5f}")
print(f"lightgbm  per-seed deltas: {[round(d, 5) for d in lgbm_deltas]}")
print(f"lightgbm  delta mean={lgbm_deltas.mean():+.5f}  std={lgbm_deltas.std():.5f}  "
      f"min={lgbm_deltas.min():+.5f}  max={lgbm_deltas.max():+.5f}")
print()
print(f"catboost  base: mean={cb_base.mean():.5f} std={cb_base.std():.5f}  "
      f"ext: mean={cb_ext.mean():.5f} std={cb_ext.std():.5f}")
print(f"catboost  per-seed deltas: {[round(d, 5) for d in cb_deltas]}")
print(f"catboost  delta mean={cb_deltas.mean():+.5f}  std={cb_deltas.std():.5f}  "
      f"min={cb_deltas.min():+.5f}  max={cb_deltas.max():+.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/matchup_extensions_seed_result.json", "w") as f:
    json.dump({
        "seeds": SEEDS,
        "lightgbm_base": lgbm_base.tolist(), "lightgbm_ext": lgbm_ext.tolist(),
        "lightgbm_delta_mean": float(lgbm_deltas.mean()), "lightgbm_delta_std": float(lgbm_deltas.std()),
        "catboost_base": cb_base.tolist(), "catboost_ext": cb_ext.tolist(),
        "catboost_delta_mean": float(cb_deltas.mean()), "catboost_delta_std": float(cb_deltas.std()),
    }, f, indent=2)
print("\nDONE")
