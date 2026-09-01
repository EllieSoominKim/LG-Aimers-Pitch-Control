"""Investigation item #3/#5: is CatBoost v3's extreme decay (0.00237, a
~75-million-x weight ratio between 2022 and 2019 TR data -- effectively
training on a single season) actually necessary for its VAL BSS, or is it
an overfit to CAL(2023)'s specific characteristics that happened to also
transfer to VAL(2024) by chance (adjacent-season luck) without being a
robust choice for a THIRD season (production's real test target, 2025)?

Retrains CatBoost with v3's exact architecture (depth=4 etc, all found by
the depth-capped search) but sweeps decay across a range from extreme to
moderate, holding everything else fixed. If VAL BSS holds up (or barely
drops) with a much more moderate decay, that's strong evidence the extreme
value was a fragile overfit to this particular CAL/VAL pairing, not a
genuinely necessary signal.
"""
import json
import sys
import time

import numpy as np
from catboost import CatBoostClassifier
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


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)

with open("C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v3.json") as f:
    v3_params = json.load(f)
fixed = {k: v3_params[k] for k in ("depth", "l2_leaf_reg", "min_data_in_leaf", "learning_rate",
                                    "random_strength", "bagging_temperature")}
print("Fixed architecture (from v3's found best trial):", fixed)

DECAYS_TO_TEST = [0.00237, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.233, 0.3]

results = []
for decay in DECAYS_TO_TEST:
    t0 = time.time()
    w_tr = decay ** (2024 - season_tr)
    n_eff = (w_tr.sum() ** 2) / (w_tr ** 2).sum()  # effective sample size (Kish's formula)
    cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                             cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                             thread_count=6, **fixed)
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
    p_cal = cb.predict_proba(X_cal_cb)[:, 1]
    p_val = cb.predict_proba(X_val_cb)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal)
    cal_val = iso.predict(p_val)
    bss_val, _, _ = brier_skill_score(y_val, cal_val, p_clim)
    print(f"decay={decay:<10g} n_eff={n_eff:>10.0f}  best_iter={cb.get_best_iteration():>4d}  "
          f"VAL BSS={bss_val:.5f}  ({time.time()-t0:.1f}s)")
    results.append({"decay": decay, "n_eff": n_eff, "best_iter": cb.get_best_iteration(), "val_bss": bss_val})

with open("C:/LG-Aimers-Pitch-Control/training/decay_sensitivity_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE")
