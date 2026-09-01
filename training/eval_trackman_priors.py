"""Item #8 (lowest priority): does adding coarse season x pitch-group
trackman environmental priors improve on matchup+extensions alone? Season-
level join (not row-order-based), so no leak-safety concern of the same
class as the matchup prior_mean bug -- each season's env values are fixed
constants computed from that season's own trackman rows only. Drop if it
doesn't help, per explicit instruction.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import CATEGORICAL_COLS, build_features, fit_category_universe, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"
TRACKMAN = "C:/LG-Aimers-Pitch-Control/data/trackman_history.csv"


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


lgbm_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json"], {})
cb_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v2.json"], {})

print("Loading train.csv + building matchup/extension features (the current best set) ...")
raw = pd.read_csv(DATA)
cats = fit_category_universe(raw)
raw = feat.add_matchup_columns_training(raw, k=30.0)
raw = feat.add_matchup_extension_columns_training(raw, k=30.0)
extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL] + feat.MATCHUP_EXTENSION_COLS

X_base = build_features(raw, categories=cats)
for col in extra_cols:
    X_base[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float64)

print("Fitting trackman env table and joining ...")
t0 = time.time()
env_table = feat.fit_trackman_env_table(TRACKMAN)
raw_env = feat.add_trackman_env_columns(raw, env_table)
X_env = X_base.copy()
for col in feat.TRACKMAN_ENV_COLS:
    X_env[col] = pd.to_numeric(raw_env[col], errors="coerce").astype(np.float64)
print(f"  done ({time.time()-t0:.1f}s), env table seasons: {env_table['season'].tolist()}")

y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values
tr_mask, cal_mask, val_mask = season <= 2022, season == 2023, season == 2024


def run(X_all, label):
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    p_clim = y_tr.mean()

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
    bss, _, _ = brier_skill_score(y_val, iso.predict(p_val), p_clim)
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
    bss, _, _ = brier_skill_score(y_val, iso.predict(p_val), p_clim)
    results["catboost"] = bss
    print(f"  catboost  VAL BSS={bss:.5f}  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")
    return results


base_results = run(X_base, "matchup + extensions (no trackman)")
env_results = run(X_env, "matchup + extensions + trackman env priors")

print(f"\n{'='*70}\nSummary\n{'='*70}")
summary = {}
for model in ("lightgbm", "catboost"):
    delta = env_results[model] - base_results[model]
    print(f"{model}: base={base_results[model]:.5f}  +trackman={env_results[model]:.5f}  delta={delta:+.5f}")
    summary[model] = {"base": base_results[model], "with_trackman": env_results[model], "delta": delta}

with open("C:/LG-Aimers-Pitch-Control/training/trackman_priors_result.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nDONE")
