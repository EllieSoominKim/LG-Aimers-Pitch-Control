"""Phase 1 follow-up item #1: rolling-origin ablation of asof_pitcher_n /
asof_batter_n, the two features the adversarial-validation pass flagged as
the dominant era-detection signal (24.4% of adversarial gain combined --
far more than the entire matchup family's 0.95%).

Also resolves item #2 (re-tuning stability): per the cost/benefit call
made in this same message, re-tuning is REVERTED to round-1's manual
hyperparameters (unstable across folds, near-zero mean effect, and a
proper 3-fold-optimized re-search would cost many hours for uncertain
payoff). So this ablation uses MANUAL hyperparameters throughout, on top
of the CONFIRMED-STABLE matchup + matchup-extensions feature set (per
rolling_origin_validation.py's findings) -- this IS the emerging candidate
pipeline, not just a diagnostic side-quest.

Three variants of asof_pitcher_n / asof_batter_n, evaluated across the
same 3 rolling folds:
  (a) raw       -- current: plain integer counts (baseline, "step 3" from
                   rolling_origin_validation.py but with MANUAL hyperparams
                   instead of tuned -- not previously measured)
  (b) dropped   -- removed entirely
  (c) log1p     -- log1p(count), compresses the high end where most of the
                   "cumulative time elapsed" signal concentrates, while
                   preserving the cold/warm ordinal distinction the model
                   plausibly needs for legitimate reliability-weighting
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"

FOLDS = [
    {"name": "fold1_2022", "tr_max": 2020, "cal": 2021, "val": 2022},
    {"name": "fold2_2023", "tr_max": 2021, "cal": 2022, "val": 2023},
    {"name": "fold3_2024", "tr_max": 2022, "cal": 2023, "val": 2024},
]

MANUAL_HP = dict(
    lgbm=dict(n_estimators=5000, learning_rate=0.02, num_leaves=15, min_child_samples=2000,
              reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7, early_stopping_rounds=100),
    cb=dict(iterations=5000, learning_rate=0.02, depth=3, l2_leaf_reg=20.0,
            min_data_in_leaf=3000, early_stopping_rounds=150),
    hgb=dict(learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=2000, l2_regularization=5.0),
    decay=0.005,
)

print("Loading train.csv + building matchup+extensions feature set ...")
t0 = time.time()
raw = pd.read_csv(DATA)
cats = fit_category_universe(raw)
raw = feat.add_matchup_columns_training(raw, k=30.0)
raw = feat.add_matchup_extension_columns_training(raw, k=30.0)
extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL] + feat.MATCHUP_EXTENSION_COLS

X_full = build_features(raw, categories=cats)
for col in extra_cols:
    X_full[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float64)

y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values
print(f"  done ({time.time()-t0:.1f}s).  X_full={X_full.shape}")

ASOF_N_COLS = ["asof_pitcher_n", "asof_batter_n"]

X_raw = X_full.copy()  # variant (a): unchanged

X_dropped = X_full.drop(columns=ASOF_N_COLS)  # variant (b)

X_log = X_full.copy()  # variant (c)
for col in ASOF_N_COLS:
    X_log[col] = np.log1p(X_log[col].clip(lower=0))


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    cols = [c for c in HGB_DROP_COLS if c in X.columns]
    return X.drop(columns=cols)


def fit_lgbm(X_tr, y_tr, w_tr, X_cal, y_cal, hp):
    m = lgb.LGBMClassifier(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                            num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"],
                            reg_lambda=hp["reg_lambda"], subsample=hp["subsample"],
                            colsample_bytree=hp["colsample_bytree"], objective="binary",
                            verbosity=-1, n_jobs=6)
    m.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
          eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(hp["early_stopping_rounds"], verbose=False)])
    return m


def fit_cb(X_tr, y_tr, w_tr, X_cal, y_cal, hp):
    m = CatBoostClassifier(iterations=hp["iterations"], learning_rate=hp["learning_rate"],
                            depth=hp["depth"], l2_leaf_reg=hp["l2_leaf_reg"],
                            min_data_in_leaf=hp["min_data_in_leaf"],
                            loss_function="Logloss", eval_metric="Logloss",
                            cat_features=CATEGORICAL_COLS, verbose=False,
                            early_stopping_rounds=hp["early_stopping_rounds"], thread_count=6)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_cal, y_cal))
    return m


def fit_hgb(X_tr, y_tr, w_tr, X_cal, y_cal, hp):
    BLOCK, PATIENCE, CAP = 25, 8, 2000
    hgb_ws = HistGradientBoostingClassifier(
        max_iter=0, learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"],
        min_samples_leaf=hp["min_samples_leaf"], l2_regularization=hp["l2_regularization"],
        categorical_features="from_dtype", early_stopping=False, warm_start=True, random_state=0)
    best_loss, best_iter, no_improve = np.inf, 0, 0
    for total in range(BLOCK, CAP + 1, BLOCK):
        hgb_ws.max_iter = total
        hgb_ws.fit(X_tr, y_tr, sample_weight=w_tr)
        loss = log_loss(y_cal, hgb_ws.predict_proba(X_cal)[:, 1])
        if loss < best_loss - 1e-6:
            best_loss, best_iter, no_improve = loss, total, 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                break
    m = HistGradientBoostingClassifier(
        max_iter=best_iter, learning_rate=hp["learning_rate"], max_leaf_nodes=hp["max_leaf_nodes"],
        min_samples_leaf=hp["min_samples_leaf"], l2_regularization=hp["l2_regularization"],
        categorical_features="from_dtype", early_stopping=False, random_state=0)
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def eval_blend(X_all, fold, hp_set):
    tr_mask = season <= fold["tr_max"]
    cal_mask = season == fold["cal"]
    val_mask = season == fold["val"]
    X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    p_clim = y_tr.mean()
    w_tr = hp_set["decay"] ** (fold["cal"] - season_tr)

    X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
    X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)

    lgbm = fit_lgbm(X_tr, y_tr, w_tr, X_cal, y_cal, hp_set["lgbm"])
    cb = fit_cb(X_tr_cb, y_tr, w_tr, X_cal_cb, y_cal, hp_set["cb"])
    hgb = fit_hgb(X_tr_hgb, y_tr, w_tr, X_cal_hgb, y_cal, hp_set["hgb"])

    models = {"lightgbm": (lgbm, X_cal, X_val), "catboost": (cb, X_cal_cb, X_val_cb), "hgb": (hgb, X_cal_hgb, X_val_hgb)}
    cal_val, per_model_bss = {}, {}
    for name, (m, Xc, Xv) in models.items():
        p_cal = m.predict_proba(Xc)[:, 1]
        p_val = m.predict_proba(Xv)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_cal, y_cal)
        cal_val[name] = iso.predict(p_val)
        per_model_bss[name], _, _ = brier_skill_score(y_val, cal_val[name], p_clim)

    avg_val = np.mean([cal_val[n] for n in ("lightgbm", "catboost", "hgb")], axis=0)
    bss_avg, _, _ = brier_skill_score(y_val, avg_val, p_clim)
    return bss_avg, per_model_bss


VARIANTS = [
    ("a_raw", X_raw),
    ("b_dropped", X_dropped),
    ("c_log1p", X_log),
]

results = {}
for fold in FOLDS:
    print(f"\n{'#'*70}\n# {fold['name']}: TR<={fold['tr_max']}  CAL={fold['cal']}  VAL={fold['val']}\n{'#'*70}")
    results[fold["name"]] = {}
    for variant_name, X_all in VARIANTS:
        t0 = time.time()
        bss, per_model = eval_blend(X_all, fold, MANUAL_HP)
        print(f"  [{variant_name}] blend VAL BSS = {bss:.5f}   per-model={per_model}   ({time.time()-t0:.1f}s)")
        results[fold["name"]][variant_name] = {"blend_bss": bss, "per_model_bss": per_model}

print(f"\n{'='*70}\nSummary table (blend VAL BSS per fold)\n{'='*70}")
header = f"{'variant':15s}" + "".join(f"{f['name']:>14s}" for f in FOLDS) + f"{'mean':>10s}{'std':>10s}"
print(header)
for variant_name, _ in VARIANTS:
    vals = [results[f["name"]][variant_name]["blend_bss"] for f in FOLDS]
    row = f"{variant_name:15s}" + "".join(f"{v:14.5f}" for v in vals) + f"{np.mean(vals):10.5f}{np.std(vals):10.5f}"
    print(row)

print(f"\n{'='*70}\nDeltas vs raw (a) per fold\n{'='*70}")
for variant_name, _ in VARIANTS[1:]:
    vals = [results[f["name"]][variant_name]["blend_bss"] - results[f["name"]]["a_raw"]["blend_bss"] for f in FOLDS]
    print(f"{variant_name:15s}" + "".join(f"{v:+14.5f}" for v in vals) + f"{np.mean(vals):+10.5f}{np.std(vals):10.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/asof_n_ablation_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE")
