import sys, time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
cats = fit_category_universe(raw)
X_all = build_features(raw, categories=cats)
y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values

def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X

tr_mask, cal_mask, val_mask = season <= 2022, season == 2023, season == 2024
X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
X_val, y_val = X_all[val_mask], y_all[val_mask]
X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)

DECAY = 0.005
w_tr = DECAY ** (2024 - season_tr)
P_CLIM = y_tr.mean()

configs = [
    dict(name="orig",   learning_rate=0.02, depth=5, l2_leaf_reg=10.0, min_data_in_leaf=2000),
    dict(name="shallow",learning_rate=0.02, depth=3, l2_leaf_reg=20.0, min_data_in_leaf=3000),
    dict(name="d4",     learning_rate=0.02, depth=4, l2_leaf_reg=20.0, min_data_in_leaf=3000),
    dict(name="lowlr",  learning_rate=0.01, depth=4, l2_leaf_reg=20.0, min_data_in_leaf=3000),
    dict(name="verylowlr", learning_rate=0.005, depth=4, l2_leaf_reg=20.0, min_data_in_leaf=3000),
]

for cfg in configs:
    name = cfg.pop("name")
    cb = CatBoostClassifier(
        iterations=5000, loss_function="Logloss", eval_metric="Logloss",
        cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=150,
        thread_count=6, **cfg,
    )
    t0 = time.time()
    cb.fit(X_tr_cb, y_tr, sample_weight=w_tr, eval_set=(X_cal_cb, y_cal))
    p = cb.predict_proba(X_val_cb)[:, 1]
    bss, bs, _ = brier_skill_score(y_val, p, P_CLIM)
    print(f"{name:12s} best_iter={cb.get_best_iteration():5d}  VAL Brier={bs:.5f}  BSS={bss:.5f}  ({time.time()-t0:.1f}s)")
