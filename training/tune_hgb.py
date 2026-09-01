"""Round 3, item #2: Optuna search for sklearn HistGradientBoostingClassifier
-- the weakest of the 3 base models, still on its original manual config.
No native optuna_integration pruning callback exists for HGB, so pruning is
done manually via the warm-start incremental-training loop already used
elsewhere in this project: report validation logloss every BLOCK
iterations, let MedianPruner decide whether to continue.
"""
import json
import sys
import time

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from tuning_common import load_split, cross_fit_calibrated_bss
from metrics import brier_skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

d = load_split(include_matchup_extensions=False)
X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
X_cal, y_cal = d["X_cal"], d["y_cal"]
X_val, y_val = d["X_val"], d["y_val"]
p_clim = d["p_clim"]

HGB_DROP_COLS = ["pitcher_id", "batter_id"]  # cardinality > 255 cap, see script.py


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
CHECKPOINT = int(sys.argv[2]) if len(sys.argv) > 2 else 120

BLOCK, PATIENCE, CAP = 25, 8, 2000


def objective(trial):
    decay = trial.suggest_float("decay", 0.0005, 0.5, log=True)
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
        max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 7, 200),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 20, 5000, log=True),
        l2_regularization=trial.suggest_float("l2_regularization", 0.01, 100.0, log=True),
    )
    w_tr = decay ** (2024 - season_tr)
    hgb = HistGradientBoostingClassifier(
        max_iter=0, warm_start=True, early_stopping=False,
        categorical_features="from_dtype", random_state=0, **params,
    )
    best_loss, best_iter, no_improve = np.inf, 0, 0
    for total in range(BLOCK, CAP + 1, BLOCK):
        hgb.max_iter = total
        hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
        loss = log_loss(y_cal, hgb.predict_proba(X_cal_hgb)[:, 1])
        # study direction="minimize" matches raw logloss's natural
        # direction directly here (no negation needed for this metric).
        trial.report(loss, total // BLOCK)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if loss < best_loss - 1e-6:
            best_loss, best_iter, no_improve = loss, total, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break

    hgb_final = HistGradientBoostingClassifier(
        max_iter=best_iter, early_stopping=False,
        categorical_features="from_dtype", random_state=0, **params,
    )
    hgb_final.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
    p_cal = hgb_final.predict_proba(X_cal_hgb)[:, 1]
    bss = cross_fit_calibrated_bss(p_cal, y_cal, p_clim)
    trial.set_user_attr("best_iteration", best_iter)
    return -bss  # study direction="minimize" -- see tune_lightgbm_v2.py's comment


t0 = time.time()
best_so_far = -np.inf
checkpoint_value = None


def log_progress(study, trial):
    global best_so_far, checkpoint_value
    true_bss = -trial.value if trial.value is not None else None
    if true_bss is not None and true_bss > best_so_far:
        best_so_far = true_bss
    n = len(study.trials)
    elapsed = time.time() - t0
    print(f"[trial {n:4d}/{N_TRIALS}] state={trial.state.name:9s} bss={true_bss} "
          f"best_so_far={best_so_far:.5f} elapsed={elapsed:.0f}s", flush=True)
    if n == CHECKPOINT:
        checkpoint_value = best_so_far
        print(f"*** CHECKPOINT at {CHECKPOINT} trials: best BSS so far = {checkpoint_value:.5f} ***", flush=True)


storage = optuna.storages.RDBStorage(url="sqlite:///C:/LG-Aimers-Pitch-Control/training/optuna_hgb.db")
study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=15, n_warmup_steps=15),
    storage=storage, study_name="hgb", load_if_exists=True,
)
study.optimize(objective, n_trials=N_TRIALS, callbacks=[log_progress])

best_value_true = -study.best_value
print(f"\nSearch done in {time.time()-t0:.1f}s over {len(study.trials)} trials")
print(f"Best CAL cross-fit-calibrated BSS (final, {N_TRIALS} trials): {best_value_true:.5f}")
print(f"Best CAL cross-fit-calibrated BSS (checkpoint, {CHECKPOINT} trials): {checkpoint_value:.5f}")
print("Best params (final):", json.dumps(study.best_params, indent=2))
print("Best iteration:", study.best_trial.user_attrs.get("best_iteration"))

best = dict(study.best_params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
best_iter = study.best_trial.user_attrs["best_iteration"]
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter, early_stopping=False,
    categorical_features="from_dtype", random_state=0, **best,
)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
raw_cal = hgb.predict_proba(X_cal_hgb)[:, 1]
raw_val = hgb.predict_proba(X_val_hgb)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, y_cal)
cal_val = iso.predict(raw_val)
bss_val_cal, bs_val_cal, _ = brier_skill_score(y_val, cal_val, p_clim)
print(f"\nVAL (untouched) CAL-fit-then-applied calibrated BSS: {bss_val_cal:.5f}  (the correct, honest number)")

with open("C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json", "w") as f:
    json.dump({"decay": decay, **best, "best_iteration": best_iter,
               "val_bss_cal_honest": bss_val_cal, "cal_bss_objective_final": best_value_true,
               "cal_bss_objective_checkpoint": checkpoint_value,
               "checkpoint_trials": CHECKPOINT, "total_trials": N_TRIALS}, f, indent=2)
print("\nSaved best_hgb_params.json")
