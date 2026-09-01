"""Same diagnostic as diagnose_decay_sensitivity.py (CatBoost), applied to
LightGBM: fixed architecture from best_lightgbm_params_v2.json, sweep decay,
retrain fresh each time, report VAL BSS + effective sample size (Kish's
formula) -- is the VAL curve flat (decay choice arbitrary/noise-driven,
like CatBoost) or genuinely peaked (decay choice is real signal)?
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

d = load_split(include_matchup_extensions=True)
X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
X_cal, y_cal = d["X_cal"], d["y_cal"]
X_val, y_val = d["X_val"], d["y_val"]
p_clim = d["p_clim"]

with open("C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json") as f:
    v2_params = json.load(f)
fixed = {k: v2_params[k] for k in ("num_leaves", "min_child_samples", "learning_rate",
                                    "reg_lambda", "reg_alpha", "subsample", "colsample_bytree")}
print("Fixed architecture (from v2's found best trial):", fixed)

DECAYS_TO_TEST = [0.0319, 0.005, 0.01, 0.02, 0.05, 0.072, 0.1, 0.15, 0.2, 0.3]

results = []
for decay in DECAYS_TO_TEST:
    t0 = time.time()
    w_tr = decay ** (2024 - season_tr)
    n_eff = (w_tr.sum() ** 2) / (w_tr ** 2).sum()
    m = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **fixed)
    m.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
          eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(80, verbose=False)])
    p_cal = m.predict_proba(X_cal)[:, 1]
    p_val = m.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    cal_val = iso.predict(p_val)
    bss_val, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    print(f"decay={decay:<10g} n_eff={n_eff:>10.0f}  best_iter={m.best_iteration_:>4d}  "
          f"VAL BSS={bss_val:.5f}  ({time.time()-t0:.1f}s)")
    results.append({"decay": decay, "n_eff": n_eff, "best_iter": m.best_iteration_, "val_bss": bss_val})

with open("C:/LG-Aimers-Pitch-Control/training/decay_sensitivity_result_lgbm.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE")
