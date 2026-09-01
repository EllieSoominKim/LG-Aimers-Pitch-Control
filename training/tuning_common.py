"""Shared data loading, splits, and objective helper for Optuna searches.

Split (matches model_comparison.py's validated protocol):
  TR  = seasons 2019-2022, sample-weighted by decay^(2024-season)
  CAL = season 2023  (Optuna's tuning target: cross-fit calibrated Brier)
  VAL = season 2024  (fully held out -- NEVER touched by Optuna, only used
                       once at the end to report the genuine improvement)

Optuna objective = cross-fit-calibrated Brier Skill Score on CAL, not raw
logloss/Brier -- this reflects what the deployed pipeline actually does
(raw model -> isotonic calibration -> score), so a trial can't "win" by
accidentally being well-calibrated on CAL without also discriminating well.
Cross-fitting (fit isotonic on half of CAL, score on the other half, both
ways) avoids the same CAL rows being used to both fit and evaluate the
calibrator, which would otherwise bias the objective optimistic.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import build_features, fit_category_universe, CATEGORICAL_COLS, TARGET_COL
from metrics import brier_skill_score

DATA = "C:/LG-Aimers-Pitch-Control/data/train.csv"

_cache = {}

# Matchup + matchup-extension numeric columns appended after CATEGORICAL_COLS
# (see script.py's comment on why order must match training exactly for
# CatBoost). Extension columns are added if present on the dataframe passed
# in (fit_matchup_extensions=True); load_split() always includes the base
# matchup_n/matchup_shrunk_success_rate pair (validated in round 2).
MATCHUP_BASE_COLS = [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL]


def load_split(include_matchup_extensions=False):
    cache_key = f"loaded_ext{include_matchup_extensions}"
    if cache_key in _cache:
        return _cache[cache_key]
    raw = pd.read_csv(DATA)
    cats = fit_category_universe(raw)

    # Matchup feature(s): expanding stats over the FULL row-ordered
    # train.csv (verified valid chronological proxy in round 2).
    raw = feat.add_matchup_columns_training(raw, k=30.0)
    extra_cols = list(MATCHUP_BASE_COLS)
    if include_matchup_extensions:
        raw = feat.add_matchup_extension_columns_training(raw, k=30.0)
        extra_cols += feat.MATCHUP_EXTENSION_COLS

    X_all = build_features(raw, categories=cats)
    for col in extra_cols:
        X_all[col] = pd.to_numeric(raw[col], errors="coerce").astype(np.float64)

    y_all = raw[TARGET_COL].values.astype(np.float64)
    season = raw["season"].values

    tr_mask = season <= 2022
    cal_mask = season == 2023
    val_mask = season == 2024

    X_tr, y_tr, season_tr = X_all[tr_mask], y_all[tr_mask], season[tr_mask]
    X_cal, y_cal = X_all[cal_mask], y_all[cal_mask]
    X_val, y_val = X_all[val_mask], y_all[val_mask]
    p_clim = y_tr.mean()  # fixed climatology reference, unweighted TR mean

    result = dict(X_tr=X_tr, y_tr=y_tr, season_tr=season_tr, X_cal=X_cal, y_cal=y_cal,
                  X_val=X_val, y_val=y_val, p_clim=p_clim, cats=cats, extra_cols=extra_cols)
    _cache[cache_key] = result
    return result


def cross_fit_calibrated_bss(p_raw, y_true, p_clim, seed=0):
    """2-fold cross-fit isotonic calibration -> honest calibrated BSS,
    without any row being used to both fit and evaluate its own calibrator."""
    rng = np.random.RandomState(seed)
    fold = rng.randint(0, 2, size=len(y_true))
    oof = np.empty_like(p_raw)
    for f in (0, 1):
        fit_mask, pred_mask = fold != f, fold == f
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p_raw[fit_mask], y_true[fit_mask])
        oof[pred_mask] = iso.predict(p_raw[pred_mask])
    bss, bs, _ = brier_skill_score(y_true, oof, p_clim)
    return bss
