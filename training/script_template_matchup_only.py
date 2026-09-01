# script.py — inference-only entry point for the LG Aimers pitch-control
# submission. Loads model/ artifacts, reads data/test.csv, writes
# output/submission.csv. Self-contained (no local imports) per the
# submission structure requirement.
#
# DELIBERATE SINGLE-CHANGE TEST (see training/train_final_matchup_only.py):
# the exact 622.14 config (fixed manual hyperparameters, uniform
# DECAY=0.005, simple-average blend of 3 globally-calibrated models) with
# ONE change -- the base pitcher-batter matchup feature added (round-3's
# add_matchup_columns_training, with the prior_mean look-ahead leak already
# fixed -- see training/verify_extension_leakage.py). No matchup
# extensions, no re-tuning, no segmented calibration, no logistic stacker.
# This isolates the matchup feature's real leaderboard effect for the
# first time -- it was previously only ever submitted bundled with round-4's
# other changes (matchup extensions + full re-tuning + segmented
# calibration), which is why we couldn't attribute the 622.14->580.75
# regression to any one specific change.
#
# Every feature is a deterministic, row-local transform of columns
# documented as available before the pitch is thrown (see
# data_description.md), OR a lookup against a static table built entirely
# from train.csv (2019-2024), which entirely precedes the 2025 test season
# -- never from other rows of the file being predicted. No cross-row
# aggregation is performed on data/test.csv.

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
# isotonic calibration breakpoints), the matchup lookup table, and one
# independent blob per model file (lightgbm.txt, hgb.joblib, catboost.cbm).
# Every one of them is tried from model/ first (case-insensitive-tolerant)
# and falls back to its embedded copy on any failure -- see the top-of-file
# comment for why.
# ---------------------------------------------------------------------

__EMBEDDED_BLOBS_PLACEHOLDER__

def _decompress_b64(b64_str):
    """gzip+base64 decode -> raw bytes. Used for every embedded blob."""
    return gzip.decompress(base64.b64decode(b64_str))


def _load_embedded_artifacts():
    """Decode the embedded categories/calibration data."""
    return json.loads(_decompress_b64(_EMBEDDED_ARTIFACTS_B64))


def _load_embedded_matchup():
    """Decode the embedded matchup lookup table."""
    return json.loads(_decompress_b64(_EMBEDDED_MATCHUP_B64))


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
# Feature engineering (mirrors training/features.py's v1 functions --
# add_engineered_columns / build_features / NUMERIC_COLS / ALL_FEATURE_COLS
# -- byte-for-byte, no matchup columns)
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

# Pitcher-batter matchup history (round-3's base feature only -- no
# extensions). Static lookup table built entirely from train.csv
# (2019-2024) -- test.csv rows never contribute to it. Column ORDER
# matters for CatBoost (fixed positional schema at training time); the
# training pipeline (train_final_matchup_only.py) appends these columns
# AFTER categories.json's categorical columns, so they must come LAST here
# too: 38 raw numeric, 4 engineered numeric, 11 categorical, then
# matchup_n, matchup_shrunk_success_rate.
MATCHUP_N_COL = "matchup_n"
MATCHUP_SHRUNK_COL = "matchup_shrunk_success_rate"
MATCHUP_NUMERIC_COLS = [MATCHUP_N_COL, MATCHUP_SHRUNK_COL]

NUMERIC_COLS = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + MATCHUP_NUMERIC_COLS
ALL_FEATURE_COLS = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS + CATEGORICAL_COLS + MATCHUP_NUMERIC_COLS

# sklearn HGB's native categorical splits cap cardinality at 255;
# pitcher_id/batter_id exceed that, so the HGB model was trained without
# them.
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


def build_features(df, categories, matchup_table):
    """categories: dict[col -> list of allowed values], learned from
    train.csv. Values not in the list become NaN (native missing-value
    handling in every model here) rather than crashing on an unseen
    category (e.g. a rookie pitcher_id never seen in training)."""
    df = add_engineered_columns(df)
    df = add_matchup_columns_inference(df, matchup_table)
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
# Model loading / calibrated blend inference
# ---------------------------------------------------------------------

def load_calibrator(path):
    """Single GLOBAL isotonic calibrator (no cold/warm segmentation --
    that was a round-4 addition, not present in the 622.14 model)."""
    npz = np.load(path)
    x, y = npz["x"], npz["y"]
    return lambda p: np.interp(p, x, y)


def _try_load(path, description, loader):
    """Load a single artifact, and on failure print the *resolved absolute
    path* it actually tried to open before re-raising."""
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
            x, y = np.array(emb["x"]), np.array(emb["y"])
            calibrators[name] = (lambda p, x=x, y=y: np.interp(p, x, y))
    return calibrators


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
    matchup_table = load_matchup_table_with_fallback(MODEL_DIR)

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
    X = build_features(test, categories, matchup_table)
    print(f" features={X.shape[1]}")

    print("Inference model...")
    if len(X):
        p_lgbm = lgbm.predict(X)
        p_cb = cb.predict_proba(catboost_frame(X))[:, 1]
        p_hgb = hgb.predict_proba(hgb_frame(X))[:, 1]

        p_lgbm = calibrators["lightgbm"](p_lgbm)
        p_cb = calibrators["catboost"](p_cb)
        p_hgb = calibrators["hgb"](p_hgb)

        # Simple average -- the blend that won the held-out 2024 comparison
        # in model_comparison.py (train_final.py's own docstring). No
        # stacker: that was a round-4 addition.
        preds = (p_lgbm + p_cb + p_hgb) / 3.0
        preds = np.clip(preds, 0.0, 1.0)
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(f"[FATAL] Unhandled exception. BASE_DIR={BASE_DIR}  cwd={os.getcwd()}")
        traceback.print_exc()
        raise
