"""Final combined validation: matchup+extensions features, v2-tuned
LightGBM, v3 size-capped CatBoost, tuned HGB, cold/warm segmented
calibration (item #5), simple single-fit stacker (item #6) -- all together,
on the CAL(2023)->VAL(2024) backtest split. This is the number that should
be quoted as "what the final production config validates to", since every
prior number this round validated pieces of this individually, never all
combined.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from features import CATEGORICAL_COLS
from tuning_common import load_split
from metrics import brier_skill_score

COLD_THRESH = 50

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


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


def load_params(paths):
    for path in paths:
        try:
            with open(path) as f:
                p = json.load(f)
            for k_ in list(p.keys()):
                if k_.startswith("val_") or k_.startswith("cal_bss") or k_ in (
                        "checkpoint_trials", "total_trials", "note", "model_size_mb"):
                    p.pop(k_, None)
            print(f"Loaded params from {path}")
            return p
        except FileNotFoundError:
            continue
    raise FileNotFoundError(paths)


lgbm_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json"])
cb_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v3.json"])
hgb_params = load_params(["C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json"])

print("\n=== Training LightGBM ===")
t0 = time.time()
lp = dict(lgbm_params)
decay_lgbm = lp.pop("decay")
w_tr_lgbm = decay_lgbm ** (2024 - season_tr)
lgbm = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **lp)
lgbm.fit(X_tr, y_tr, sample_weight=w_tr_lgbm, categorical_feature=CATEGORICAL_COLS,
         eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
         callbacks=[lgb.early_stopping(80, verbose=False)])
print(f"  best_iter={lgbm.best_iteration_}  ({time.time()-t0:.1f}s)")

print("\n=== Training CatBoost (size-capped v3) ===")
t0 = time.time()
cp = dict(cb_params)
decay_cb = cp.pop("decay")
w_tr_cb = decay_cb ** (2024 - season_tr)
X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
cb = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                         cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                         thread_count=6, **cp)
cb.fit(X_tr_cb, y_tr, sample_weight=w_tr_cb, eval_set=(X_cal_cb, y_cal))
print(f"  best_iter={cb.get_best_iteration()}  ({time.time()-t0:.1f}s)")

print("\n=== Training sklearn HGB ===")
t0 = time.time()
hp = dict(hgb_params)
decay_hgb = hp.pop("decay")
best_iter_hgb = hp.pop("best_iteration")
w_tr_hgb = decay_hgb ** (2024 - season_tr)
X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)
hgb = HistGradientBoostingClassifier(
    max_iter=best_iter_hgb, categorical_features="from_dtype", early_stopping=False,
    random_state=0, **hp)
hgb.fit(X_tr_hgb, y_tr, sample_weight=w_tr_hgb)
print(f"  max_iter={best_iter_hgb}  ({time.time()-t0:.1f}s)")

models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}
names = list(models.keys())

raw_cal = {name: m.predict_proba(Xc)[:, 1] for name, (m, Xc, Xv) in models.items()}
raw_val = {name: m.predict_proba(Xv)[:, 1] for name, (m, Xc, Xv) in models.items()}

cold_mask_cal = (X_cal["asof_pitcher_n"] < COLD_THRESH).values
cold_mask_val = (X_val["asof_pitcher_n"] < COLD_THRESH).values
print(f"\nCAL cold rows: {cold_mask_cal.sum()}/{len(cold_mask_cal)}  "
      f"VAL cold rows: {cold_mask_val.sum()}/{len(cold_mask_val)}")

print("\n=== Cold/warm segmented calibration (item #5) + individual VAL BSS ===")
cal_cal_seg, cal_val_seg = {}, {}
for name in names:
    p_cal, p_val = raw_cal[name], raw_val[name]
    iso_cold = IsotonicRegression(out_of_bounds="clip")
    iso_cold.fit(p_cal[cold_mask_cal], y_cal[cold_mask_cal])
    iso_warm = IsotonicRegression(out_of_bounds="clip")
    iso_warm.fit(p_cal[~cold_mask_cal], y_cal[~cold_mask_cal])

    seg_cal = np.empty_like(p_cal)
    seg_cal[cold_mask_cal] = iso_cold.predict(p_cal[cold_mask_cal])
    seg_cal[~cold_mask_cal] = iso_warm.predict(p_cal[~cold_mask_cal])
    cal_cal_seg[name] = seg_cal

    seg_val = np.empty_like(p_val)
    seg_val[cold_mask_val] = iso_cold.predict(p_val[cold_mask_val])
    seg_val[~cold_mask_val] = iso_warm.predict(p_val[~cold_mask_val])
    cal_val_seg[name] = seg_val

    bss, _, _ = brier_skill_score(y_val, seg_val, p_clim)
    print(f"  {name} (cold/warm calibrated): VAL BSS = {bss:.5f}")

print("\n=== Simple single-fit stacker (item #6) ===")
stack_X_cal = np.column_stack([cal_cal_seg[n] for n in names])
stack_X_val = np.column_stack([cal_val_seg[n] for n in names])
stacker = LogisticRegression()
stacker.fit(stack_X_cal, y_cal)
stack_val_pred = stacker.predict_proba(stack_X_val)[:, 1]
bss_final, bs_final, _ = brier_skill_score(y_val, stack_val_pred, p_clim)
print(f"Stacker coefficients: {dict(zip(names, stacker.coef_[0]))}  intercept: {stacker.intercept_[0]}")
print(f"\n{'='*70}")
print(f"FINAL COMBINED CONFIG: VAL (untouched, 2024) BSS = {bss_final:.5f}  (Brier = {bs_final:.5f})")
print(f"{'='*70}")

# For reference: simple average blend too
avg_val_pred = stack_X_val.mean(axis=1)
bss_avg, _, _ = brier_skill_score(y_val, avg_val_pred, p_clim)
print(f"(reference) simple-average blend: VAL BSS = {bss_avg:.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/final_combined_result.json", "w") as f:
    json.dump({"names": names, "val_bss_final_stacked": bss_final,
               "val_bss_simple_average": bss_avg,
               "per_model_val_bss": {n: brier_skill_score(y_val, cal_val_seg[n], p_clim)[0] for n in names}},
              f, indent=2)
print("\nDONE")
