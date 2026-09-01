"""Item #2 wrap-up: HGB's Optuna search was stopped early (trial 84/400,
after a 41-trial plateau at best CAL-objective BSS=0.00977) per user
decision, rather than waiting ~25h more for a search that had stopped
improving. Load the best trial from the persisted SQLite study, retrain
once with those params, and compute the honest CAL-fit-then-VAL-apply BSS
-- same final step tune_hgb.py itself would have done at N_TRIALS.
"""
import json
import sys

import numpy as np
import optuna
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from tuning_common import load_split
from metrics import brier_skill_score

d = load_split(include_matchup_extensions=False)
X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
X_cal, y_cal = d["X_cal"], d["y_cal"]
X_val, y_val = d["X_val"], d["y_val"]
p_clim = d["p_clim"]

HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)

storage = optuna.storages.RDBStorage(url="sqlite:///C:/LG-Aimers-Pitch-Control/training/optuna_hgb.db")
study = optuna.load_study(study_name="hgb", storage=storage)

completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
print(f"Loaded study: {len(study.trials)} total trials, {len(completed)} completed")
best_trial = min(completed, key=lambda t: t.value)  # direction=minimize, value=-bss
best_value_true = -best_trial.value
print(f"Best trial: #{best_trial.number}  CAL-objective BSS={best_value_true:.5f}")
print("Best params:", json.dumps(best_trial.params, indent=2))
print("Best iteration:", best_trial.user_attrs.get("best_iteration"))

best = dict(best_trial.params)
decay = best.pop("decay")
w_tr = decay ** (2024 - season_tr)
best_iter = best_trial.user_attrs["best_iteration"]

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
    json.dump({
        "decay": decay, **best, "best_iteration": best_iter,
        "val_bss_cal_honest": bss_val_cal,
        "cal_bss_objective_final": best_value_true,
        "cal_bss_objective_checkpoint": None,
        "checkpoint_trials": None,
        "total_trials": len(completed),
        "note": "search stopped early at trial 84/400 after 41-trial plateau; this is the best trial found, not a full 400-trial result",
    }, f, indent=2)
print("\nSaved best_hgb_params.json")
