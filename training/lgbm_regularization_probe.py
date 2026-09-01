import sys, time
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
cats = fit_category_universe(raw)
X_all = build_features(raw, categories=cats)
y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values

tr_mask, cal_mask, val_mask = season <= 2022, season == 2023, season == 2024
X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
p_clim = y_tr.mean()
DECAY = 0.15
w_tr = DECAY ** (2024 - season_tr)

configs = [
    dict(name="orig",        learning_rate=0.03, num_leaves=63,  min_child_samples=200,  reg_lambda=1.0,  subsample=0.8, colsample_bytree=0.8),
    dict(name="shallow_reg", learning_rate=0.02, num_leaves=15,  min_child_samples=2000, reg_lambda=5.0,  subsample=0.7, colsample_bytree=0.7),
    dict(name="mid_reg",     learning_rate=0.02, num_leaves=31,  min_child_samples=1000, reg_lambda=3.0,  subsample=0.7, colsample_bytree=0.7),
    dict(name="lowlr",       learning_rate=0.005, num_leaves=31, min_child_samples=1000, reg_lambda=3.0,  subsample=0.7, colsample_bytree=0.7),
    dict(name="stumps",      learning_rate=0.02, num_leaves=7,   min_child_samples=3000, reg_lambda=5.0,  subsample=0.7, colsample_bytree=0.7),
]

for cfg in configs:
    name = cfg.pop("name")
    m = lgb.LGBMClassifier(n_estimators=5000, objective="binary", verbosity=-1, n_jobs=6, **cfg)
    t0 = time.time()
    m.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
          eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    p = m.predict_proba(X_val)[:, 1]
    bss, bs, _ = brier_skill_score(y_val, p, p_clim)
    print(f"{name:12s} best_iter={m.best_iteration_:5d}  VAL Brier={bs:.5f}  BSS={bss:.5f}  ({time.time()-t0:.1f}s)")
