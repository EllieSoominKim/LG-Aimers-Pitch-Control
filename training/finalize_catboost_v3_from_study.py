"""tune_catboost_v3.py's search completed all 151 trials successfully
(best CAL-objective BSS=0.01046), but a cosmetic bug in the resume
invocation (checkpoint_value stayed None because the resumed session's
trial count never crossed the checkpoint threshold) crashed the script
before it saved best_catboost_params_v3.json. Load the best trial from the
persisted SQLite study directly, retrain once, and save the params +
honest VAL BSS + model size -- same final step tune_catboost_v3.py itself
would have done.
"""
import json
import os
import sys
import tempfile

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

d = load_split(include_matchup_extensions=True)
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

storage = optuna.storages.RDBStorage(url="sqlite:///C:/LG-Aimers-Pitch-Control/training/optuna_catboost_v3.db")
study = optuna.load_study(study_name="catboost_v3", storage=storage)
completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Loaded study: {len(study.trials)} total trials, {len(completed)} completed")
best_trial = min(completed, key=lambda t: t.value)
best_value_true = -best_trial.value
print(f"Best trial: #{best_trial.number}  CAL-objective BSS={best_value_true:.5f}")
print("Best params:", json.dumps(best_trial.params, indent=2))

best = dict(best_trial.params)
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
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_cal, y_cal)
cal_val = iso.predict(raw_val)
bss_val_cal, bs_val_cal, _ = brier_skill_score(y_val, cal_val, p_clim)
print(f"\nVAL (untouched) CAL-fit-then-applied calibrated BSS: {bss_val_cal:.5f}  (the correct, honest number)")

with tempfile.TemporaryDirectory() as tmpdir:
    tmp_path = os.path.join(tmpdir, "catboost_probe.cbm")
    cb.save_model(tmp_path)
    size_mb = os.path.getsize(tmp_path) / 1e6
print(f"Trained model size: {size_mb:.1f} MB  (depth cap was 8)")

with open("C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v3.json", "w") as f:
    json.dump({"decay": decay, **best, "val_bss_cal_honest": bss_val_cal,
               "cal_bss_objective_final": best_value_true,
               "total_trials": len(completed), "model_size_mb": size_mb,
               "note": "finalized from persisted study after a cosmetic crash in tune_catboost_v3.py's resume invocation"},
              f, indent=2)
print("\nSaved best_catboost_params_v3.json")
