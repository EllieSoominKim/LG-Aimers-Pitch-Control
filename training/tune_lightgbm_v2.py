"""Round 3, items #1+#3: Optuna search (LightGBM) with the matchup feature
now included, 400 trials with a MedianPruner + native LightGBM pruning
callback, persistent SQLite storage (so progress survives/can be
monitored), and per-trial logging (running best score after every trial --
the lack of this in round 2 made a long CatBoost run impossible to
distinguish from "stuck").

Reports the best-at-~120-trials checkpoint (apples-to-apples budget vs the
round-2 search) separately from the best-at-400-trials final result, to
cleanly separate "does re-tuning with matchup help" (#1) from "does more
search budget help" (#3).
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import optuna
from optuna_integration import LightGBMPruningCallback

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

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CHECKPOINT = int(sys.argv[2]) if len(sys.argv) > 2 else 120


def objective(trial):
    decay = trial.suggest_float("decay", 0.0005, 0.5, log=True)
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 7, 200),
        min_child_samples=trial.suggest_int("min_child_samples", 30, 5000, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.003, 0.15, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 0.005, 100.0, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 0.005, 100.0, log=True),
        subsample=trial.suggest_float("subsample", 0.4, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
    )
    w_tr = decay ** (2024 - season_tr)
    pruning_cb = LightGBMPruningCallback(trial, "binary_logloss", valid_name="valid_0")
    model = lgb.LGBMClassifier(
        n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **params
    )
    model.fit(
        X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
        eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(80, verbose=False), pruning_cb],
    )
    p_cal = model.predict_proba(X_cal)[:, 1]
    bss = cross_fit_calibrated_bss(p_cal, y_cal, p_clim)
    trial.set_user_attr("best_iteration", model.best_iteration_)
    # Study direction is "minimize" (see below) so the intermediate
    # binary_logloss values LightGBMPruningCallback reports (naturally
    # lower-is-better) stay direction-consistent with the study -- optuna
    # refuses to prune otherwise ("intermediate values are inconsistent
    # with the objective values"). Negate BSS here; un-negate on report.
    return -bss


t0 = time.time()
best_so_far = -np.inf  # tracked in true (un-negated) BSS terms
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


storage = optuna.storages.RDBStorage(url="sqlite:///C:/LG-Aimers-Pitch-Control/training/optuna_lightgbm_v2.db")
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=30),
    storage=storage, study_name="lightgbm_v2", load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_progress])

best_value_true = -study.best_value
print(f"\nSearch done in {time.time()-t0:.1f}s over {len(study.trials)} trials")
print(f"Best CAL cross-fit-calibrated BSS (final, {N_TRIALS} trials): {best_value_true:.5f}")
print(f"Best CAL cross-fit-calibrated BSS (checkpoint, {CHECKPOINT} trials): {checkpoint_value:.5f}")
print("Best params (final):", json.dumps(study.best_params, indent=2))

# Refit best (final) config on TR, evaluate on the untouched VAL fold.
best = dict(study.best_params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
model = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **best)
model.fit(
    X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
    eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
    callbacks=[lgb.early_stopping(80, verbose=False)],
)
raw_cal = model.predict_proba(X_cal)[:, 1]
raw_val = model.predict_proba(X_val)[:, 1]
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, y_cal)
cal_val = iso.predict(raw_val)
bss_val_cal, bs_val_cal, _ = brier_skill_score(y_val, cal_val, p_clim)
print(f"\nVAL (untouched) CAL-fit-then-applied calibrated BSS: {bss_val_cal:.5f}  (the correct, honest number)")

with open("C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json", "w") as f:
    json.dump({"decay": decay, **best, "val_bss_cal_honest": bss_val_cal,
               "cal_bss_objective_final": best_value_true,
               "cal_bss_objective_checkpoint": checkpoint_value,
               "checkpoint_trials": CHECKPOINT, "total_trials": N_TRIALS}, f, indent=2)
print("\nSaved best_lightgbm_params_v2.json")
