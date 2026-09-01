"""Phase 1 diagnostic item #3: adversarial validation. Train a classifier
to distinguish 2024 rows from 2019-2023 rows using the full current
feature set (base features + matchup + matchup extensions -- everything
under suspicion). High AUC means the feature set as a whole encodes "which
era is this row from" rather than being purely row-local pre-pitch
information; high per-feature importance on specific columns flags exactly
which ones are the leakiest era-detectors.
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"

print("Loading train.csv + building full feature set (base + matchup + extensions) ...")
t0 = time.time()
raw = pd.read_csv(DATA)
cats = fit_category_universe(raw)
raw = feat.add_matchup_columns_training(raw, k=30.0)
raw = feat.add_matchup_extension_columns_training(raw, k=30.0)
extra_cols = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL] + feat.MATCHUP_EXTENSION_COLS

X_all = build_features(raw, categories=cats)
for col in extra_cols:
    X_all[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float64)

season = raw["season"].values
is_2024 = (season == 2024).astype(int)

# Drop "season" itself -- it's literally the column that defines the label
# here (is_2024 = season==2024), so leaving it in makes the whole exercise
# degenerate (any model gets ~100% AUC by reading it off directly). The
# real question is whether OTHER (engineered/matchup) features carry excess
# era-specific signal beyond the legitimate, known "season" indicator.
X_all = X_all.drop(columns=["season"])  # "season" is RAW_NUMERIC_COLS, not categorical -- CATEGORICAL_COLS unaffected
print(f"  done ({time.time()-t0:.1f}s).  X={X_all.shape} (season dropped)  2024 rows={is_2024.sum()}  other rows={(1-is_2024).sum()}")

X_train, X_test, y_train, y_test = train_test_split(
    X_all, is_2024, test_size=0.2, random_state=0, stratify=is_2024)
print(f"adv-train={X_train.shape}  adv-test={X_test.shape}")

print("\n=== Training adversarial classifier (LightGBM) ===")
t0 = time.time()
m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                        min_child_samples=200, objective="binary", verbosity=-1, n_jobs=6)
# Use a slice of train as internal eval for early stopping (not the adv-test set itself).
X_fit, X_es, y_fit, y_es = train_test_split(X_train, y_train, test_size=0.1, random_state=1, stratify=y_train)
m.fit(X_fit, y_fit, categorical_feature=CATEGORICAL_COLS,
      eval_set=[(X_es, y_es)], eval_metric="auc",
      callbacks=[lgb.early_stopping(30, verbose=False)])
print(f"  best_iter={m.best_iteration_}  ({time.time()-t0:.1f}s)")

p_test = m.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, p_test)
print(f"\nAdversarial AUC (2024 vs 2019-2023, held-out adv-test): {auc:.4f}")
print("(0.5 = indistinguishable / no era leakage.  1.0 = perfectly separable.)")

names = X_all.columns.tolist()
gains = m.booster_.feature_importance(importance_type="gain")
total_gain = gains.sum()
rows = sorted(zip(names, gains), key=lambda r: -r[1])
print(f"\nTop 20 features distinguishing 2024 from 2019-2023 (by gain):")
for n, g in rows[:20]:
    flag = "  <-- MATCHUP FAMILY" if ("matchup" in n or n == "pitcher_vs_hand_shrunk_success_rate") else ""
    print(f"  {n:45s} gain={g:12.1f} ({100*g/total_gain:5.2f}%){flag}")

print(f"\n=== Repeat with base features only (no matchup/extensions) for comparison ===")
X_base = build_features(raw, categories=cats).drop(columns=["season"])
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_base, is_2024, test_size=0.2, random_state=0, stratify=is_2024)
X_fit_b, X_es_b, y_fit_b, y_es_b = train_test_split(X_train_b, y_train_b, test_size=0.1, random_state=1, stratify=y_train_b)
m_base = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             min_child_samples=200, objective="binary", verbosity=-1, n_jobs=6)
m_base.fit(X_fit_b, y_fit_b, categorical_feature=CATEGORICAL_COLS,
           eval_set=[(X_es_b, y_es_b)], eval_metric="auc",
           callbacks=[lgb.early_stopping(30, verbose=False)])
p_test_b = m_base.predict_proba(X_test_b)[:, 1]
auc_base = roc_auc_score(y_test_b, p_test_b)
print(f"Adversarial AUC (base features only, no matchup): {auc_base:.4f}")
print(f"Delta from adding matchup+extensions: {auc - auc_base:+.4f}")

import json
with open("C:/LG-Aimers-Pitch-Control/training/adversarial_validation_result.json", "w") as f:
    json.dump({
        "auc_full_feature_set": float(auc),
        "auc_base_only": float(auc_base),
        "top_features": [(n, float(g), float(g/total_gain)) for n, g in rows[:20]],
    }, f, indent=2)
print("\nDONE")
