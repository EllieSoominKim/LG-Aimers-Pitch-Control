"""Optuna search (CatBoost) optimizing cross-fit-calibrated Brier Skill
Score on the CAL fold (season 2023). VAL (season 2024) stays untouched
until the final report.
"""
import json
import sys
import time

import numpy as np
import optuna
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split, cross_fit_calibrated_bss
from metrics import brier_skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

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


X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def objective(trial):
    decay = trial.suggest_float("decay", 0.0005, 0.5, log=True)
    params = dict(
        depth=trial.suggest_int("depth", 2, 8),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 50.0, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 50, 5000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 5.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 5.0),
    )
    w_tr = decay ** (2024 - season_tr)
    cb = CatBoostClassifier(
        iterations=3000, loss_function="Logloss", eval_metric="Logloss",
        cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
        thread_count=6, **params,
    )
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
    p_cal = cb.predict_proba(X_cal_cb)[:, 1]
    bss = cross_fit_calibrated_bss(p_cal, y_cal, p_clim)
    trial.set_user_attr("best_iteration", cb.get_best_iteration())
    return bss


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Search done in {time.time()-t0:.1f}s over {len(study.trials)} trials")
print(f"Best CAL cross-fit-calibrated BSS: {study.best_value:.5f}")
print("Best params:", json.dumps(study.best_params, indent=2))

best = dict(study.best_params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
cb = CatBoostClassifier(
    iterations=3000, loss_function="Logloss", eval_metric="Logloss",
    cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
    thread_count=6, **best,
)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
p_val_raw = cb.predict_proba(X_val_cb)[:, 1]
bss_val_raw, bs_val_raw, _ = brier_skill_score(y_val, p_val_raw, p_clim)
bss_val_cal = cross_fit_calibrated_bss(p_val_raw, y_val, p_clim)
print(f"\nVAL (untouched) raw BSS:                {bss_val_raw:.5f}")
print(f"VAL (untouched) cross-fit-calibrated BSS: {bss_val_cal:.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/best_catboost_params.json", "w") as f:
    json.dump({"decay": decay, **best, "val_bss_raw": bss_val_raw, "val_bss_cal": bss_val_cal,
               "cal_bss_objective": study.best_value}, f, indent=2)
print("\nSaved best_catboost_params.json")
