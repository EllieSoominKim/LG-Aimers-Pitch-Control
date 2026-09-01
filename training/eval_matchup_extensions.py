"""Item #4: does adding the matchup extension columns (last-N rolling trend,
pitcher-vs-batter-hand shrunk rate, matchup x pitcher/batter rate
interactions) improve on the base matchup feature (matchup_n +
matchup_shrunk_success_rate) alone? Trains LightGBM and CatBoost with the
newly-tuned v2 params (falls back to v1/manual if v2 missing) on both
feature sets, reports honest CAL-fit-then-VAL-apply BSS for each.
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
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in ("checkpoint_trials", "total_trials", "best_iteration", "note"):
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


def run(include_extensions):
    label = "matchup + extensions" if include_extensions else "matchup base only"
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    d = load_split(include_matchup_extensions=include_extensions)
    X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
    X_cal, y_cal = d["X_cal"], d["y_cal"]
    X_val, y_val = d["X_val"], d["y_val"]
    p_clim = d["p_clim"]

    results = {}

    t0 = time.time()
    lp = dict(lgbm_params)
    decay_lgbm = lp.pop("decay")
    w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
    lgbm = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **lp)
    lgbm.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
             eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
             callbacks=[lgb.early_stopping(80, verbose=False)])
    p_cal, p_val = lgbm.predict_proba(X_cal)[:, 1], lgbm.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    cal_val = iso.predict(p_val)
    bss, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    results["lightgbm"] = bss
    print(f"  lightgbm  VAL BSS={bss:.5f}  best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    cp = dict(cb_params)
    decay_cb = cp.pop("decay")
    w_tr_cb = decay_cb ** (2024 - season_tr)
    X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
    cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                             cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                             thread_count=6, **cp)
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
    p_cal, p_val = cb.predict_proba(X_cal_cb)[:, 1], cb.predict_proba(X_val_cb)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    cal_val = iso.predict(p_val)
    bss, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    results["catboost"] = bss
    print(f"  catboost  VAL BSS={bss:.5f}  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

    return results


base_results = run(False)
ext_results = run(True)

print(f"\n{'='*70}\nSummary\n{'='*70}")
summary = {}
for model in ("lightgbm", "catboost"):
    delta = ext_results[model] - base_results[model]
    print(f"{model}: base={base_results[model]:.5f}  +extensions={ext_results[model]:.5f}  delta={delta:+.5f}")
    summary[model] = {"base": base_results[model], "with_extensions": ext_results[model], "delta": delta}

with open("C:/LG-Aimers-Pitch-Control/training/matchup_extensions_result.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE")
