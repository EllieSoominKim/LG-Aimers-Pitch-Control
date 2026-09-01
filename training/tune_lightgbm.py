"""Optuna search (LightGBM) optimizing cross-fit-calibrated Brier Skill
Score on the CAL fold (season 2023). VAL (season 2024) stays untouched
until the final report.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna

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

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 120


def objective(trial):
    decay = trial.suggest_float("decay", 0.0005, 0.5, log=True)
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 7, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 50, 5000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.01, 50.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.01, 50.0, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
    )
    w_tr = decay ** (2024 - season_tr)
    model = lgb.LGBMClassifier(
        n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **params
    )
    model.fit(
        X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
        eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    p_cal = model.predict_proba(X_cal)[:, 1]
    bss = cross_fit_calibrated_bss(p_cal, y_cal, p_clim)
    trial.set_user_attr("best_iteration", model.best_iteration_)
    return bss


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
t0 = time.time()
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
print(f"Search done in {time.time()-t0:.1f}s over {len(study.trials)} trials")
print(f"Best CAL cross-fit-calibrated BSS: {study.best_value:.5f}")
print("Best params:", json.dumps(study.best_params, indent=2))

# Refit best config on TR, evaluate on the untouched VAL fold for the
# genuine held-out comparison number.
best = dict(study.best_params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
model = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **best)
model.fit(
    X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
    eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(80, verbose=False)],
)
p_val_raw = model.predict_proba(X_val)[:, 1]
bss_val_raw, bs_val_raw, _ = brier_skill_score(y_val, p_val_raw, p_clim)
bss_val_cal = cross_fit_calibrated_bss(p_val_raw, y_val, p_clim)
print(f"\nVAL (untouched) raw BSS:                {bss_val_raw:.5f}")
print(f"VAL (untouched) cross-fit-calibrated BSS: {bss_val_cal:.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params.json", "w") as f:
    json.dump({"decay": decay, **best, "val_bss_raw": bss_val_raw, "val_bss_cal": bss_val_cal,
               "cal_bss_objective": study.best_value}, f, indent=2)
print("\nSaved best_lightgbm_params.json")
