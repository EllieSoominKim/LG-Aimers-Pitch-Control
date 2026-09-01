"""Phase 1 diagnostic item #1: rolling-origin validation. Re-evaluates the
project's actual sequence of decisions -- (0) 622.14 baseline -> (1) +base
matchup feature -> (2) +per-model re-tuning (with matchup present) -> (3)
+matchup extensions (with tuned hyperparams) -- across THREE rolling folds
instead of just the single 2023->2024 fold everything was previously
validated on:

  fold 1: TR=2019-2020, CAL=2021, VAL=2022
  fold 2: TR=2019-2021, CAL=2022, VAL=2023
  fold 3: TR=2019-2022, CAL=2023, VAL=2024  (the ONLY fold used until now)

Each step's hyperparameters/blend are held FIXED at whatever was actually
found/deployed (no per-fold re-tuning -- that's out of scope for this
diagnostic pass and would cost 300+ Optuna trials x 3 folds x 3 models).
The question this answers: do the specific hyperparameter values and
feature-engineering decisions we already made on the 2023->2024 fold
generalize to other fold boundaries, or were they overfit to that one
fold's specific noise?

Simple-average blend of 3 globally-calibrated models throughout (matching
train_final.py; no stacker, no segmented calibration -- keeping the
comparison uncontaminated by those separate round-4 additions).
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
    {"name": "fold3_2024", "tr_max": 2022, "cal": 2023, "val": 2024},  # the original, only-ever-used fold
]

MANUAL_HP = dict(
    lgbm=dict(n_estimators=5000, learning_rate=0.02, num_leaves=15, min_child_samples=2000,
              reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7, early_stopping_rounds=100),
    cb=dict(iterations=5000, learning_rate=0.02, depth=3, l2_leaf_reg=20.0,
            min_data_in_leaf=3000, early_stopping_rounds=150),
    hgb=dict(learning_rate=0.02, max_leaf_nodes=15, min_samples_leaf=2000, l2_regularization=5.0),
    decay=0.005,
)


def load_json_params(path):
    with open(path) as f:
        p = json.load(f)
    for k in list(p.keys()):
        if k.startswith("val_") or k.startswith("cal_bss") or k in (
                "checkpoint_trials", "total_trials", "note", "model_size_mb"):
            p.pop(k, None)
    return p


TUNED_LGBM = load_json_params("C:/LG-Aimers-Pitch-Control/training/best_lightgbm_params_v2.json")
TUNED_CB = load_json_params("C:/LG-Aimers-Pitch-Control/training/best_catboost_params_v3.json")
TUNED_HGB = load_json_params("C:/LG-Aimers-Pitch-Control/training/best_hgb_params.json")

print("Loading train.csv + building feature variants ...")
t0 = time.time()
raw = pd.read_csv(DATA)
cats = fit_category_universe(raw)

# Feature variant 0/1: no matchup / base matchup (leak-fixed)
raw_m = feat.add_matchup_columns_training(raw.copy(), k=30.0)
raw_me = feat.add_matchup_extension_columns_training(raw_m.copy(), k=30.0)

X_base = build_features(raw, categories=cats)

X_matchup = X_base.copy()
for col in (feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL):
    X_matchup[col] = pd.to_numeric(raw_m[col], errors="coerce").astype(np.float64)

X_matchup_ext = X_matchup.copy()
for col in feat.MATCHUP_EXTENSION_COLS:
    X_matchup_ext[col] = pd.to_numeric(raw_me[col], errors="coerce").astype(np.float64)

y_all = raw[TARGET_COL].values.astype(np.float64)
season = raw["season"].values
print(f"  done ({time.time()-t0:.1f}s).  X_base={X_base.shape}  X_matchup={X_matchup.shape}  X_matchup_ext={X_matchup_ext.shape}")


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


def fit_lgbm(X_tr, y_tr, w_tr, X_cal, y_cal, hp, manual):
    if manual:
        m = lgb.LGBMClassifier(n_estimators=hp["n_estimators"], learning_rate=hp["learning_rate"],
                                num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"],
                                reg_lambda=hp["reg_lambda"], subsample=hp["subsample"],
                                colsample_bytree=hp["colsample_bytree"], objective="binary",
                                verbosity=-1, n_jobs=6)
        es = hp["early_stopping_rounds"]
    else:
        p = dict(hp)
        p.pop("decay", None)
        m = lgb.LGBMClassifier(n_estimators=3000, objective="binary", verbosity=-1, n_jobs=6, **p)
        es = 80
    m.fit(X_tr, y_tr, sample_weight=w_tr, categorical_feature=CATEGORICAL_COLS,
          eval_set=[(X_cal, y_cal)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(es, verbose=False)])
    return m


def fit_cb(X_tr, y_tr, w_tr, X_cal, y_cal, hp, manual):
    if manual:
        m = CatBoostClassifier(iterations=hp["iterations"], learning_rate=hp["learning_rate"],
                                depth=hp["depth"], l2_leaf_reg=hp["l2_leaf_reg"],
                                min_data_in_leaf=hp["min_data_in_leaf"],
                                loss_function="Logloss", eval_metric="Logloss",
                                cat_features=CATEGORICAL_COLS, verbose=False,
                                early_stopping_rounds=hp["early_stopping_rounds"], thread_count=6)
    else:
        p = dict(hp)
        p.pop("decay", None)
        m = CatBoostClassifier(iterations=3000, loss_function="Logloss", eval_metric="Logloss",
                                cat_features=CATEGORICAL_COLS, verbose=False, early_stopping_rounds=80,
                                thread_count=6, **p)
    m.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=(X_cal, y_cal))
    return m


def fit_hgb(X_tr, y_tr, w_tr, X_cal, y_cal, hp, manual):
    if manual:
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
    else:
        p = dict(hp)
        p.pop("decay", None)
        best_iter = p.pop("best_iteration")
        m = HistGradientBoostingClassifier(max_iter=best_iter, categorical_features="from_dtype",
                                            early_stopping=False, random_state=0, **p)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def eval_blend(X_all, fold, hp_set, manual):
    tr_mask = season <= fold["tr_max"]
    cal_mask = season == fold["cal"]
    val_mask = season == fold["val"]
    X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    p_clim = y_tr.mean()

    decay = hp_set["decay"] if manual else hp_set["decay"]
    w_tr = decay ** (fold["cal"] - season_tr)

    X_tr_cb, X_cal_cb, X_val_cb = catboost_frame(X_tr), catboost_frame(X_cal), catboost_frame(X_val)
    X_tr_hgb, X_cal_hgb, X_val_hgb = hgb_frame(X_tr), hgb_frame(X_cal), hgb_frame(X_val)

    lgbm = fit_lgbm(X_tr, y_tr, w_tr, X_cal, y_cal, hp_set["lgbm"] if manual else TUNED_LGBM, manual)
    cb = fit_cb(X_tr_cb, y_tr, w_tr, X_cal_cb, y_cal, hp_set["cb"] if manual else TUNED_CB, manual)
    hgb = fit_hgb(X_tr_hgb, y_tr, w_tr, X_cal_hgb, y_cal, hp_set["hgb"] if manual else TUNED_HGB, manual)

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


STEPS = [
    ("0_no_matchup_manual", X_base, MANUAL_HP, True),
    ("1_matchup_manual", X_matchup, MANUAL_HP, True),
    ("2_matchup_tuned", X_matchup, MANUAL_HP, False),
    ("3_matchup_ext_tuned", X_matchup_ext, MANUAL_HP, False),
]

results = {}
for fold in FOLDS:
    print(f"\n{'#'*70}\n# {fold['name']}: TR<={fold['tr_max']}  CAL={fold['cal']}  VAL={fold['val']}\n{'#'*70}")
    results[fold["name"]] = {}
    for step_name, X_all, hp_set, manual in STEPS:
        t0 = time.time()
        bss, per_model = eval_blend(X_all, fold, hp_set, manual)
        print(f"  [{step_name}] blend VAL BSS = {bss:.5f}   per-model={per_model}   ({time.time()-t0:.1f}s)")
        results[fold["name"]][step_name] = {"blend_bss": bss, "per_model_bss": per_model}

print(f"\n{'='*70}\nSummary table (blend VAL BSS per fold)\n{'='*70}")
step_names = [s[0] for s in STEPS]
header = f"{'step':30s}" + "".join(f"{f['name']:>14s}" for f in FOLDS) + f"{'mean':>10s}{'std':>10s}"
print(header)
for step_name in step_names:
    vals = [results[f["name"]][step_name]["blend_bss"] for f in FOLDS]
    row = f"{step_name:30s}" + "".join(f"{v:14.5f}" for v in vals) + f"{np.mean(vals):10.5f}{np.std(vals):10.5f}"
    print(row)

print(f"\n{'='*70}\nIncremental deltas per fold\n{'='*70}")
delta_defs = [
    ("matchup effect (0->1)", "0_no_matchup_manual", "1_matchup_manual"),
    ("tuning effect (1->2)", "1_matchup_manual", "2_matchup_tuned"),
    ("extensions effect (2->3)", "2_matchup_tuned", "3_matchup_ext_tuned"),
]
for label, a, b in delta_defs:
    vals = [results[f["name"]][b]["blend_bss"] - results[f["name"]][a]["blend_bss"] for f in FOLDS]
    print(f"{label:30s}" + "".join(f"{v:+14.5f}" for v in vals) + f"{np.mean(vals):+10.5f}{np.std(vals):10.5f}")

with open("C:/LG-Aimers-Pitch-Control/training/rolling_origin_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nDONE")
