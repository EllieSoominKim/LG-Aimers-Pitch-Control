"""Round 3, items #1+#3: Optuna search (CatBoost) with the matchup feature
now included, 400 trials with a MedianPruner + native CatBoost pruning
callback, persistent SQLite storage, per-trial logging.
"""
import json
import sys
import time

import numpy as np
import optuna
from catboost import CatBoostClassifier
from optuna_integration import CatBoostPruningCallback

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split, cross_fit_calibrated_bss
from metrics import brier_skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

d = load_split(include_matchup_extensions=False)
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

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CHECKPOINT = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def objective(trial):
    decay = trial.suggest_float("decay", 0.0005, 0.5, log=True)
    params = dict(
        depth=trial.suggest_int("depth", 2, 10),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 100.0, log=True),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 10, 5000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 10.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
    )
    w_tr = decay ** (2024 - season_tr)
    pruning_cb = CatBoostPruningCallback(trial, "Logloss")
    cb = CatBoostClassifier(
        iterations=3000, loss_function="Logloss", eval_metric="Logloss",
        cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
        thread_count=6, **params,
    )
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal), callbacks=[pruning_cb])
    pruning_cb.check_pruned()  # CatBoost callbacks can't raise directly; must check after fit
    p_cal = cb.predict_proba(X_cal_cb)[:, 1]
    bss = cross_fit_calibrated_bss(p_cal, y_cal, p_clim)
    trial.set_user_attr("best_iteration", cb.get_best_iteration())
    return -bss  # study direction="minimize" -- see tune_lightgbm_v2.py's comment on why


t0 = time.time()
best_so_far = -np.inf
checkpoint_value = None
checkpoint_params = None


def log_progress(study, trial):
    global best_so_far, checkpoint_value, checkpoint_params
    true_bss = -trial.value if trial.value is not None else None
    if true_bss is not None and true_bss > best_so_far:
        best_so_far = true_bss
    n = len(study.trials)
    elapsed = time.time() - t0
    state = trial.state.name
    print(f"[trial {n:4d}/{N_TRIALS}] state={state:9s} bss={true_bss} "
          f"best_so_far={best_so_far:.5f} elapsed={elapsed:.0f}s", flush=True)
    if n == CHECKPOINT:
        checkpoint_value = best_so_far
        checkpoint_params = dict(study.best_params)
        print(f"*** CHECKPOINT at {CHECKPOINT} trials: best BSS so far = {checkpoint_value:.5f} ***", flush=True)


storage = optuna.storages.RDBStorage(url="sqlite:///C:/LG-Aimers-Pitch-Control/training/optuna_catboost_v2.db")
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=30),
    storage=storage, study_name="catboost_v2", load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_progress])

best_value_true = -study.best_value
print(f"\nSearch done in {time.time()-t0:.1f}s over {len(study.trials)} trials")
print(f"Best CAL cross-fit-calibrated BSS (final, {N_TRIALS} trials): {best_value_true:.5f}")
print(f"Best CAL cross-fit-calibrated BSS (checkpoint, {CHECKPOINT} trials): {checkpoint_value:.5f}")
print("Best params (final):", json.dumps(study.best_params, indent=2))

best = dict(study.best_params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
cb = CatBoostClassifier(
    iterations=3000, loss_function="Logloss", eval_metric="Logloss",
    cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
    thread_count=6, **best,
)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
raw_cal = cb.predict_proba(X_cal_cb)[:, 1]
raw_val = cb.predict_proba(X_val_cb)[:, 1]
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, y_cal)
cal_val = iso.predict(raw_val)
bss_val_cal, bs_val_cal, _ = brier_skill_score(y_val, cal_val, p_clim)
print(f"\nVAL (untouched) CAL-fit-then-applied calibrated BSS: {bss_val_cal:.5f}  (the correct, honest number)")

with open("C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v2.json", "w") as f:
    json.dump({"decay": decay, **best, "val_bss_cal_honest": bss_val_cal,
               "cal_bss_objective_final": best_value_true,
               "cal_bss_objective_checkpoint": checkpoint_value,
               "checkpoint_trials": CHECKPOINT, "total_trials": N_TRIALS}, f, indent=2)
print("\nSaved best_catboost_params_v2.json")
