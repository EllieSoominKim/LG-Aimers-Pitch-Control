"""Quick grid search over the recency-weight decay using LightGBM only.

TR = seasons 2019-2022 (weighted by decay^(2024-season))
CAL = season 2023 (early-stopping eval set)
VAL = season 2024 (held out, final comparison metric — untouched otherwise)
"""

import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"

print("Loading train.csv ...")
t0 = time.time()
raw = pd.read_csv(DATA)
print(f"  loaded {raw.shape} in {time.time()-t0:.1f}s")

cats = fit_category_universe(raw)
X_all = build_features(raw, categories=cats)
y_all = raw[TARGET_COL].values
season = raw["season"].values

tr_mask = season <= 2022
cal_mask = season == 2023
val_mask = season == 2024

X_tr, y_tr = X_all[tr_mask], y_all[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
season_tr = season[tr_mask]

print(f"TR={X_tr.shape}  CAL={X_cal.shape}  VAL={X_val.shape}")

p_clim = y_tr.mean()  # naive reference forecast: base rate of TR only
print(f"Climatology (TR base rate) = {p_clim:.4f}")

results = []
for decay in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    w_tr = decay ** (2024 - season_tr)
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=200,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        verbosity=-1,
        n_jobs=6,
    )
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        categorical_feature=CATEGORICAL_COLS,
        eval_set=[(X_cal, y_cal)],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    p_val = model.predict_proba(X_val)[:, 1]
    bss, bs, bs_ref = brier_skill_score(y_val, p_val, p_clim)
    dt = time.time() - t0
    print(f"decay={decay:.2f}  best_iter={model.best_iteration_:5d}  "
          f"VAL Brier={bs:.5f}  BSS={bss:.5f}  ({dt:.1f}s)")
    results.append((decay, model.best_iteration_, bs, bss))

print()
best = max(results, key=lambda r: r[3])
print(f"Best decay: {best[0]} (BSS={best[3]:.5f})")
