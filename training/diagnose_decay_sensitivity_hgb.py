"""Same diagnostic, applied to HGB: fixed architecture from
best_hgb_params.json, sweep decay, re-run the warm-start early-stopping
loop fresh for each decay (best_iteration legitimately depends on decay,
unlike the fixed-iteration CatBoost/LightGBM sweeps), report VAL BSS +
effective sample size.
"""
import json
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from tuning_common import load_split
from metrics import brier_skill_score

d = load_split(include_matchup_extensions=True)
X_tr, y_tr, season_tr = d["X_tr"], d["y_tr"], d["season_tr"]
X_cal, y_cal = d["X_cal"], d["y_cal"]
X_val, y_val = d["X_val"], d["y_val"]
p_clim = d["p_clim"]

HGB_DROP_COLS = ["pitcher_id", "batter_id"]
X_tr_hgb, X_cal_hgb, X_val_hgb = (X_tr.drop(columns=HGB_DROP_COLS), X_cal.drop(columns=HGB_DROP_COLS),
                                   X_val.drop(columns=HGB_DROP_COLS))

with open("C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json") as f:
    hgb_params = json.load(f)
fixed = {k: hgb_params[k] for k in ("learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
print("Fixed architecture (from tuned best trial):", fixed)

DECAYS_TO_TEST = [0.005, 0.05, 0.15, 0.384, 0.5]
FIXED_ITER = 25  # per user decision: skip per-decay warm-start rediscovery (too slow,
                 # HGB is the weakest ensemble model) -- use the already-tuned
                 # best_iteration for all points, single fast .fit() per point,
                 # matching the CatBoost/LightGBM sweep methodology. Less rigorous
                 # (iteration count could genuinely shift with decay) but gets a
                 # usable flat-vs-peaked read in minutes instead of hours.

results = []
for decay in DECAYS_TO_TEST:
    t0 = time.time()
    w_tr = decay ** (2024 - season_tr)
    n_eff = (w_tr.sum() ** 2) / (w_tr ** 2).sum()
    best_iter = FIXED_ITER
    hgb = HistGradientBoostingClassifier(
        max_iter=best_iter, early_stopping=False, categorical_features="from_dtype",
        random_state=0, **fixed)
    hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr)
    p_cal = hgb.predict_proba(X_cal_hgb)[:, 1]
    p_val = hgb.predict_proba(X_val_hgb)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    cal_val = iso.predict(p_val)
    bss_val, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    print(f"decay={decay:<10g} n_eff={n_eff:>10.0f}  best_iter={best_iter:>4d}  "
          f"VAL BSS={bss_val:.5f}  ({time.time()-t0:.1f}s)")
    results.append({"decay": decay, "n_eff": n_eff, "best_iter": best_iter, "val_bss": bss_val})

with open("C:/LG-Aimers-Pitch-Control/training/decay_sensitivity_result_hgb.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE")
