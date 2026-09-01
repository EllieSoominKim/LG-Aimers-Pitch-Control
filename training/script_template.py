# script.py — inference-only entry point for the LG Aimers pitch-control
# submission. Loads model/ artifacts, reads data/test.csv, writes
# output/submission.csv. Self-contained (no local imports) per the
# submission structure requirement.
#
# Model (v3): logistic-stack blend of 3 cold/warm-segment-calibrated
# gradient-boosted trees (LightGBM, CatBoost -- both Optuna-tuned over 400
# pruned trials each against cross-fit-calibrated Brier Skill Score on a
# 2023 holdout fold, matchup-feature-aware -- and sklearn
# HistGradientBoostingClassifier, Optuna-tuned), trained on train.csv
# seasons 2019-2023 with recency-decay sample weighting (per-model decay,
# also Optuna-tuned) and calibrated on season 2024. Adds a pitcher-batter
# matchup-history feature plus round-4 extensions (last-3-meetings rolling
# trend, pitcher-vs-batter-hand shrunk rate, matchup-rate x own-asof-rate
# interactions) on top of the original feature set -- validated robust
# across 5 random seeds (LightGBM +0.00105, CatBoost +0.00080 mean VAL BSS
# delta) after fixing a look-ahead bug in the shrinkage-prior constant used
# by both the base matchup feature and its extensions (see
# training/verify_extension_leakage.py; the fix did not erode the gains --
# CatBoost's extension delta nearly tripled once corrected). Calibration is
# cold/warm segmented (asof_pitcher_n<50 gets its own isotonic calibrator)
# -- a small, consistent gain found by testing global vs. segmented
# calibration; a leverage(li)-bucket segmentation was tried too and
# rejected (net negative for CatBoost). A Bayesian-shrinkage variant of the
# asof_* rate columns, a coarse trackman environmental prior, and
# multi-seed base-model ensembling were all tried and dropped (measurably
# hurt, or not worth the added complexity/artifact size for the return) --
# see training/eval_result_*.json and training/*_result.json for the full
# comparisons.
#
# Every feature is a deterministic, row-local transform of columns
# documented as available before the pitch is thrown (see
# data_description.md), OR a lookup against a static table built entirely
# from train.csv (2019-2024), which entirely precedes the 2025 test season
# -- never from other rows of the file being predicted. No cross-row
# aggregation is performed on data/test.csv.
#
# EVERY model/ artifact (categories.json, calibration_*.npz, lightgbm.txt,
# catboost.cbm, hgb.joblib, matchup_lookup.json, matchup_extension_lookup.json,
# stacker.json) is ALSO embedded directly in this file's source (gzip+base64)
# and used as a fallback if the real model/ file is absent for any reason.
# This is why the file is large (~16MB) -- two submissions in a row failed
# with FileNotFoundError on a model/ file that was independently verified
# present, correctly named, and byte-identical/loadable in the submitted
# zip, pointing at a systematic issue with model/ file-staging on the eval
# server rather than anything wrong with the zip. Embedding removes the
# dependency entirely. See training/generate_embedded_artifacts.py and
# training/train_final_v3.py to regenerate after a retrain.

import base64
import gzip
import io
import json
import os
import traceback

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

ID_COL = "row_id"
TARGET_COL = "control_success"

# Anchor every path in this script to the script's own location, never to
# the process's current working directory (a prior submission crashed with
# FileNotFoundError on "./model/categories.json" because the eval harness
# invoked this script from a working directory that was NOT the directory
# script.py lives in).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Embedded artifacts, gzip-compressed + base64-encoded, baked directly into
# this file's source: the small auxiliary blob (category vocabulary,
# isotonic calibration breakpoints, stacker weights), the matchup lookup
# table, and one independent blob per large model file (lightgbm.txt,
# hgb.joblib, catboost.cbm). Every one of them is tried from model/ first
# (case-insensitive-tolerant) and falls back to its embedded copy on any
# failure -- see the top-of-file comment for why.
# ---------------------------------------------------------------------

__EMBEDDED_BLOBS_PLACEHOLDER__

def _decompress_b64(b64_str):
    """gzip+base64 decode -> raw bytes. Used for every embedded blob."""
    return gzip.decompress(base64.b64decode(b64_str))


def _load_embedded_artifacts():
    """Decode the embedded categories/calibration/stacker data (~20KB)."""
    return json.loads(_decompress_b64(_EMBEDDED_ARTIFACTS_B64))


def _load_embedded_matchup():
    """Decode the embedded matchup lookup table (~5MB raw, ~950KB blob)."""
    return json.loads(_decompress_b64(_EMBEDDED_MATCHUP_B64))


def _load_embedded_matchup_ext():
    """Decode the embedded matchup extension lookup table (round 4, item #4)."""
    return json.loads(_decompress_b64(_EMBEDDED_MATCHUP_EXT_B64))


_EMBEDDED = _load_embedded_artifacts()


def _resolve_artifact_path(directory, expected_name):
    """Return the path to open for `expected_name` inside `directory`.

    Tries the exact-case path first. If that's absent but a
    case-INsensitive match exists (Windows dev/test hides case mismatches
    that Linux eval servers would reject), uses that actual on-disk name
    instead and prints a warning. If nothing matches, returns the
    originally-expected path unchanged so the caller's own error still
    fires with a clear message.
    """
    exact = os.path.join(directory, expected_name)
    if os.path.isfile(exact):
        return exact
    if os.path.isdir(directory):
        for actual in os.listdir(directory):
            if actual.lower() == expected_name.lower() and actual != expected_name:
                print(f"[WARN] case mismatch under {directory}: expected "
                      f"'{expected_name}', found '{actual}' -- using it")
                return os.path.join(directory, actual)
    return exact

# ---------------------------------------------------------------------
# Feature engineering (mirrors training/features.py)
# ---------------------------------------------------------------------

RAW_CATEGORICAL_COLS = [
    "pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
    "pitcher_hand", "batter_hand", "top_bottom", "game_type", "base_state",
]
ENGINEERED_CATEGORICAL_COLS = ["count_state", "hand_matchup"]
CATEGORICAL_COLS = RAW_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

RAW_NUMERIC_COLS = [
    "season", "game_month", "game_dayofweek", "inning",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
ENGINEERED_NUMERIC_COLS = [
    "pitcher_batter_form", "pitcher_momentum_success",
    "pitcher_momentum_middle", "is_dp_situation",
]
# Pitcher-batter matchup history + round-4 extensions (see
# training/features.py's add_matchup_columns_training /
# add_matchup_extension_columns_training / fit_matchup_lookup /
# fit_matchup_extension_lookup / MATCHUP_* constants). All of these come
# from static lookup tables built entirely from train.csv (2019-2024) --
# test.csv rows never contribute to them (no cross-row aggregation on the
# file being predicted).
#
# Column ORDER matters here: the trained CatBoost model bakes in a fixed
# feature layout at training time and does NOT realign by name at predict
# time (LightGBM does, silently -- CatBoost errors instead). The training
# pipeline (train_final_v3.py) appends these columns via plain DataFrame
# assignment AFTER categories.json's categorical columns were already in
# place, so they must come LAST here too, in this exact order, matching
# CatBoostClassifier().feature_names_ exactly: 42 numeric, then 11
# categorical, then matchup_n, matchup_shrunk_success_rate,
# matchup_last3_success_rate, pitcher_vs_hand_shrunk_success_rate,
# matchup_x_pitcher_rate, matchup_x_batter_rate.
MATCHUP_N_COL = "matchup_n"
MATCHUP_SHRUNK_COL = "matchup_shrunk_success_rate"
MATCHUP_LAST_N = 3
MATCHUP_LASTN_COL = f"matchup_last{MATCHUP_LAST_N}_success_rate"
MATCHUP_HAND_COL = "pitcher_vs_hand_shrunk_success_rate"
MATCHUP_INTERACTION_PITCHER_COL = "matchup_x_pitcher_rate"
MATCHUP_INTERACTION_BATTER_COL = "matchup_x_batter_rate"
MATCHUP_NUMERIC_COLS = [
    MATCHUP_N_COL, MATCHUP_SHRUNK_COL,
    MATCHUP_LASTN_COL, MATCHUP_HAND_COL,
    MATCHUP_INTERACTION_PITCHER_COL, MATCHUP_INTERACTION_BATTER_COL,
]

# Cold-start threshold for segmented calibration (item #5): rows with fewer
# than this many prior pitches by this pitcher get their own isotonic
# calibrator, fit separately from the "warm" majority -- a small but
# consistent VAL BSS gain (+0.00009 to +0.00011) found by testing global vs.
# segmented calibration on cold-start vs. warm and on leverage (li) terciles;
# li-bucket segmentation was tested too and rejected (net negative for
# CatBoost). See training/investigate_slice_calibration.py.
COLD_THRESH = 50

NUMERIC_COLS = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + MATCHUP_NUMERIC_COLS
ALL_FEATURE_COLS = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + CATEGORICAL_COLS + MATCHUP_NUMERIC_COLS

# sklearn HGB's native categorical splits cap cardinality at 255;
# pitcher_id/batter_id exceed that, so the HGB model was trained without
# them (team_id/hand/asof_*/matchup already carry the generalizable signal).
HGB_DROP_COLS = ["pitcher_id", "batter_id"]


def add_engineered_columns(df):
    df = df.copy()
    df["count_state"] = (
        df["balls_before"].astype(int).astype(str) + "-"
        + df["strikes_before"].astype(int).astype(str)
    )
    df["hand_matchup"] = (
        df["pitcher_hand"].astype(int).astype(str) + "-"
        + df["batter_hand"].astype(int).astype(str)
    )
    df["pitcher_batter_form"] = (
        df["asof_pitcher_success_rate"] * df["asof_batter_success_rate"]
    )
    df["pitcher_momentum_success"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )
    df["pitcher_momentum_middle"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )
    df["is_dp_situation"] = (
        (df["runner_on_1b"] == 1) & (df["outs_before"] < 2)
    ).astype(int)
    return df


def add_matchup_columns_inference(df, matchup_table):
    """Static (pitcher_id, batter_id) -> (n, shrunk_rate) lookup, built
    entirely from train.csv. Pairs never seen in training get the global
    prior mean and n=0 -- the same "unknown category" fallback philosophy
    used everywhere else in this script."""
    df = df.copy()
    prior_mean = matchup_table["prior_mean"]
    pairs = matchup_table["pairs"]
    keys = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)
    n_vals, rate_vals = [], []
    for k_ in keys:
        entry = pairs.get(k_)
        if entry is None:
            n_vals.append(0.0)
            rate_vals.append(prior_mean)
        else:
            n_vals.append(float(entry["n"]))
            rate_vals.append(float(entry["rate"]))
    df[MATCHUP_N_COL] = n_vals
    df[MATCHUP_SHRUNK_COL] = rate_vals
    return df


def add_matchup_extension_columns_inference(df, ext_table):
    """Static lookups for the round-4 matchup extension columns, built
    entirely from train.csv (2019-2024). Requires MATCHUP_SHRUNK_COL
    already present (add_matchup_columns_inference must run first) --
    the interaction terms are computed directly from already-loaded
    features, no lookup needed for those two."""
    df = df.copy()
    prior_mean = ext_table["prior_mean"]
    pair_lastn = ext_table["pair_lastn"]
    hand_lookup = ext_table["pitcher_hand"]

    pair_keys = df["pitcher_id"].astype(str) + "_" + df["batter_id"].astype(str)
    df[MATCHUP_LASTN_COL] = [pair_lastn.get(k_, prior_mean) for k_ in pair_keys]

    hand_keys = df["pitcher_id"].astype(str) + "_" + df["batter_hand"].astype(str)
    df[MATCHUP_HAND_COL] = [hand_lookup.get(k_, prior_mean) for k_ in hand_keys]

    df[MATCHUP_INTERACTION_PITCHER_COL] = df[MATCHUP_SHRUNK_COL] * df["asof_pitcher_success_rate"].fillna(prior_mean)
    df[MATCHUP_INTERACTION_BATTER_COL] = df[MATCHUP_SHRUNK_COL] * df["asof_batter_success_rate"].fillna(prior_mean)
    return df


def build_features(df, categories, matchup_table, ext_table):
    """categories: dict[col -> list of allowed values], learned from
    train.csv. Values not in the list become NaN (native missing-value
    handling in every model here) rather than crashing on an unseen
    category (e.g. a rookie pitcher_id never seen in training)."""
    df = add_engineered_columns(df)
    df = add_matchup_columns_inference(df, matchup_table)
    df = add_matchup_extension_columns_inference(df, ext_table)
    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    for col in CATEGORICAL_COLS:
        X[col] = pd.Categorical(df[col], categories=categories[col])
    return X[ALL_FEATURE_COLS]


def catboost_frame(X):
    X = X.copy()
    for c in CATEGORICAL_COLS:
        X[c] = X[c].astype(object).where(X[c].notna(), "__MISSING__").astype(str)
    return X


def hgb_frame(X):
    return X.drop(columns=HGB_DROP_COLS)


# ---------------------------------------------------------------------
# Data / submission I/O
# ---------------------------------------------------------------------

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(
            f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: {list(df.columns)}")
    return df


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# ---------------------------------------------------------------------
# Model loading / calibrated stacked inference
# ---------------------------------------------------------------------

def load_calibrator(path):
    """Cold/warm segmented calibrator (item #5): returns a dict with
    "cold" and "warm" interpolation functions, applied per-row based on
    asof_pitcher_n < COLD_THRESH at prediction time."""
    npz = np.load(path)
    x_cold, y_cold = npz["x_cold"], npz["y_cold"]
    x_warm, y_warm = npz["x_warm"], npz["y_warm"]
    return {
        "cold": lambda p: np.interp(p, x_cold, y_cold),
        "warm": lambda p: np.interp(p, x_warm, y_warm),
    }


def apply_segmented_calibration(p_raw, calibrator, cold_mask):
    """calibrator: dict with "cold"/"warm" interpolation functions (from
    load_calibrator or the embedded fallback below). cold_mask: boolean
    array, True where asof_pitcher_n < COLD_THRESH."""
    out = np.empty_like(p_raw, dtype=np.float64)
    out[cold_mask] = calibrator["cold"](p_raw[cold_mask])
    out[~cold_mask] = calibrator["warm"](p_raw[~cold_mask])
    return out


def _try_load(path, description, loader):
    """Load a single artifact, and on failure print the *resolved absolute
    path* it actually tried to open before re-raising. This is what makes a
    future path problem diagnosable from the error output alone, instead of
    a bare FileNotFoundError with no context about which path or why."""
    abs_path = os.path.abspath(path)
    print(f"  loading {description}: {abs_path}")
    try:
        return loader(path)
    except Exception:
        print(f"[FATAL] Failed to load {description}.")
        print(f"        resolved absolute path: {abs_path}")
        print(f"        path exists: {os.path.exists(abs_path)}")
        raise


def load_categories_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "categories.json")
    try:
        print(f"  loading category vocabulary: {os.path.abspath(path)}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the embedded category vocabulary baked into script.py.")
        return _EMBEDDED["categories"]


def load_calibrators_with_fallback(model_dir):
    calibrators = {}
    for name in ("lightgbm", "catboost", "hgb"):
        path = _resolve_artifact_path(model_dir, f"calibration_{name}.npz")
        try:
            print(f"  loading {name} calibrator: {os.path.abspath(path)}")
            calibrators[name] = load_calibrator(path)
        except Exception as e:
            print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
                  f"falling back to the embedded {name} calibration breakpoints.")
            emb = _EMBEDDED["calibration"][name]
            xc, yc = np.array(emb["x_cold"]), np.array(emb["y_cold"])
            xw, yw = np.array(emb["x_warm"]), np.array(emb["y_warm"])
            calibrators[name] = {
                "cold": (lambda p, x=xc, y=yc: np.interp(p, x, y)),
                "warm": (lambda p, x=xw, y=yw: np.interp(p, x, y)),
            }
    return calibrators


def load_stacker_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "stacker.json")
    try:
        print(f"  loading stacker weights: {os.path.abspath(path)}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the embedded stacker weights.")
        return _EMBEDDED["stacker"]


def load_matchup_table_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "matchup_lookup.json")
    try:
        print(f"  loading matchup lookup table: {os.path.abspath(path)}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the embedded matchup lookup table.")
        return _load_embedded_matchup()


def load_matchup_extension_table_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "matchup_extension_lookup.json")
    try:
        print(f"  loading matchup extension lookup table: {os.path.abspath(path)}")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the embedded matchup extension lookup table.")
        return _load_embedded_matchup_ext()


def load_lightgbm_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "lightgbm.txt")
    try:
        print(f"  loading LightGBM model: {os.path.abspath(path)}")
        return lgb.Booster(model_file=path)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the LightGBM model embedded in script.py.")
        model_str = _decompress_b64(_EMBEDDED_LIGHTGBM_B64).decode("utf-8")
        return lgb.Booster(model_str=model_str)


def load_catboost_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "catboost.cbm")
    cb = CatBoostClassifier()
    try:
        print(f"  loading CatBoost model: {os.path.abspath(path)}")
        cb.load_model(path)
        return cb
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the CatBoost model embedded in script.py.")
        blob = _decompress_b64(_EMBEDDED_CATBOOST_B64)
        cb.load_model(blob=blob)
        return cb


def load_hgb_with_fallback(model_dir):
    path = _resolve_artifact_path(model_dir, "hgb.joblib")
    try:
        print(f"  loading HistGradientBoosting model: {os.path.abspath(path)}")
        return joblib.load(path)
    except Exception as e:
        print(f"[WARN] Could not load {os.path.abspath(path)} ({e!r}); "
              f"falling back to the HGB model embedded in script.py.")
        raw = _decompress_b64(_EMBEDDED_HGB_B64)
        return joblib.load(io.BytesIO(raw))


def stacker_blend(preds_by_name, stacker):
    """Logistic-regression stack: sigmoid(sum(coef_i * pred_i) + intercept).
    stacker = {"names": [...], "coef": [...], "intercept": float}."""
    z = np.full(len(next(iter(preds_by_name.values()))), stacker["intercept"], dtype=np.float64)
    for name, coef in zip(stacker["names"], stacker["coef"]):
        z += coef * preds_by_name[name]
    return 1.0 / (1.0 + np.exp(-z))


def main():
    TEST_DIR = os.path.join(BASE_DIR, "data")
    MODEL_DIR = os.path.join(BASE_DIR, "model")
    OUT_DIR = os.path.join(BASE_DIR, "output")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print(f"BASE_DIR   = {BASE_DIR}")
    print(f"cwd        = {os.getcwd()}  (unused for path resolution, logged for diagnostics only)")
    print(f"TEST_DIR   = {TEST_DIR}")
    print(f"MODEL_DIR  = {MODEL_DIR}")
    print(f"OUT_DIR    = {OUT_DIR}")
    if os.path.isdir(MODEL_DIR):
        print(f"MODEL_DIR contents: {sorted(os.listdir(MODEL_DIR))}")
    else:
        print(f"[WARNING] MODEL_DIR does not exist at this resolved path.")
    if os.path.isdir(TEST_DIR):
        print(f"TEST_DIR contents: {sorted(os.listdir(TEST_DIR))}")
    else:
        print(f"[WARNING] TEST_DIR does not exist at this resolved path.")

    print("Load models...")
    categories = load_categories_with_fallback(MODEL_DIR)
    calibrators = load_calibrators_with_fallback(MODEL_DIR)
    stacker = load_stacker_with_fallback(MODEL_DIR)
    matchup_table = load_matchup_table_with_fallback(MODEL_DIR)
    ext_table = load_matchup_extension_table_with_fallback(MODEL_DIR)

    lgbm = load_lightgbm_with_fallback(MODEL_DIR)
    cb = load_catboost_with_fallback(MODEL_DIR)
    hgb = load_hgb_with_fallback(MODEL_DIR)
    print(" OK")

    print("Load test data...")
    test = _try_load(_resolve_artifact_path(TEST_DIR, "test.csv"), "test.csv", load_test)
    sub = _try_load(
        _resolve_artifact_path(TEST_DIR, "sample_submission.csv"),
        "sample_submission.csv", load_sample_submission)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, categories, matchup_table, ext_table)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    if len(X):
        # Cold-start segment for calibration (item #5): asof_pitcher_n is
        # still a plain feature column in X at this point (dropped only
        # from HGB's own input frame, further down), so it's read directly
        # off the already-built feature matrix.
        cold_mask = (X["asof_pitcher_n"] < COLD_THRESH).values

        p_lgbm = lgbm.predict(X)
        p_cb = cb.predict_proba(catboost_frame(X))[:, 1]
        p_hgb = hgb.predict_proba(hgb_frame(X))[:, 1]

        p_lgbm = apply_segmented_calibration(p_lgbm, calibrators["lightgbm"], cold_mask)
        p_cb = apply_segmented_calibration(p_cb, calibrators["catboost"], cold_mask)
        p_hgb = apply_segmented_calibration(p_hgb, calibrators["hgb"], cold_mask)

        preds = stacker_blend({"lightgbm": p_lgbm, "catboost": p_cb, "hgb": p_hgb}, stacker)
        preds = np.clip(preds, 0.0, 1.0)
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    # Never let this fail with a bare, hard-to-diagnose trace: log the full
    # traceback plus BASE_DIR context, then re-raise so the process still
    # exits non-zero (a failed run must fail loudly, not silently continue).
    try:
        main()
    except Exception:
        print(f"[FATAL] Unhandled exception. BASE_DIR={BASE_DIR}  cwd={os.getcwd()}")
        traceback.print_exc()
        raise
