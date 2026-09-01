"""Shared feature engineering for the LG Aimers pitch-control model.

This module is the single source of truth for feature engineering during
local training/CV. Its logic is mirrored verbatim (kept in sync by hand)
inside the final submission's script.py, which must stay a single
self-contained file per the competition's submission structure.

Only information available *before* the pitch is used — every input column
comes straight from train.csv/test.csv (all documented as pre-pitch) plus
deterministic transforms of those columns. Nothing here looks at other rows
of the dataset being transformed (no target encoding, no cross-row stats),
so the same function is safe to apply to test.csv one row/batch at a time.
"""

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

# Raw columns treated as nominal categories (tree-native categorical split,
# not one-hot). IDs are included here (not as numeric) since their integer
# values carry no ordinal meaning.
RAW_CATEGORICAL_COLS = [
    "pitcher_id",
    "batter_id",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "top_bottom",
    "game_type",
    "base_state",
]

# Engineered categorical columns (added by build_features).
ENGINEERED_CATEGORICAL_COLS = [
    "count_state",
    "hand_matchup",
]

CATEGORICAL_COLS = RAW_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS

# Raw numeric columns passed through as-is.
RAW_NUMERIC_COLS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

# Engineered numeric columns (added by build_features).
ENGINEERED_NUMERIC_COLS = [
    "pitcher_batter_form",
    "pitcher_momentum_success",
    "pitcher_momentum_middle",
    "is_dp_situation",
]

NUMERIC_COLS = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS


def add_engineered_columns(df):
    """Add engineered feature columns in place-safe fashion, return new df.

    Every input is either a raw pre-pitch column or a deterministic
    row-local transform of raw pre-pitch columns. No cross-row aggregation.
    """
    df = df.copy()

    # Full ball-strike count as its own categorical (pitcher approach differs
    # by exact count, not just balls-strikes leverage direction).
    df["count_state"] = (
        df["balls_before"].astype(int).astype(str)
        + "-"
        + df["strikes_before"].astype(int).astype(str)
    )

    # Handedness matchup interaction.
    df["hand_matchup"] = (
        df["pitcher_hand"].astype(int).astype(str)
        + "-"
        + df["batter_hand"].astype(int).astype(str)
    )

    # Pitcher/batter recent-form interaction.
    df["pitcher_batter_form"] = (
        df["asof_pitcher_success_rate"] * df["asof_batter_success_rate"]
    )

    # Momentum: recent (prev1) vs longer-window (prev5) form. Positive =
    # trending up recently relative to the last-5-games baseline.
    df["pitcher_momentum_success"] = (
        df["asof_pitcher_prev1_game_success_rate"]
        - df["asof_pitcher_prev5_game_success_rate"]
    )
    df["pitcher_momentum_middle"] = (
        df["asof_pitcher_prev1_game_middle_rate"]
        - df["asof_pitcher_prev5_game_middle_rate"]
    )

    # Classic double-play situational flag: runner on 1st, fewer than 2 outs.
    df["is_dp_situation"] = (
        (df["runner_on_1b"] == 1) & (df["outs_before"] < 2)
    ).astype(int)

    return df


def build_features(df, categories=None):
    """Build the model-ready feature frame.

    Parameters
    ----------
    df : raw input dataframe (train.csv or test.csv schema, row_id present,
        control_success optional).
    categories : dict[str, list] or None
        Fixed category universes (col -> sorted list of allowed values) to
        apply via pd.Categorical, learned once from the training data. Any
        value not in the list becomes NaN (i.e. "unknown category"), which
        every candidate model treats as a native missing-value branch. If
        None, categories are inferred from `df` itself (used only when
        *fitting* the category universe on the training set).

    Returns
    -------
    X : DataFrame with NUMERIC_COLS as float and CATEGORICAL_COLS as
        pandas 'category' dtype, column order = ALL_FEATURE_COLS.
    """
    df = add_engineered_columns(df)

    X = pd.DataFrame(index=df.index)
    for col in NUMERIC_COLS:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)

    for col in CATEGORICAL_COLS:
        vals = df[col]
        if categories is not None:
            X[col] = pd.Categorical(vals, categories=categories[col])
        else:
            X[col] = pd.Categorical(vals)

    return X[ALL_FEATURE_COLS]


def fit_category_universe(df):
    """Learn the fixed category universe from a (training) dataframe."""
    df = add_engineered_columns(df)
    return {col: sorted(df[col].dropna().unique().tolist()) for col in CATEGORICAL_COLS}


# =======================================================================
# v2 additions: reliability-weighted (Bayesian-shrinkage) asof_* features
# and pitcher-batter matchup history. Kept separate from the v1 functions
# above (which remain the currently-deployed baseline) so the two feature
# sets can be A/B compared cleanly on the held-out VAL fold before either
# is adopted.
# =======================================================================

# (rate column, sample-size column) pairs eligible for shrinkage -- every
# asof_* rate column that has a companion n column. The prev1/3/5-game rates
# have no n column (they're per-game aggregates of unknown pitch count) so
# they're left as-is; unlike them, all of these are cumulative-since-start
# rates paired with an exact count of the samples behind them.
SHRINKAGE_SPECS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]
SHRUNK_COLS = [f"shrunk_{rate_col}" for rate_col, _ in SHRINKAGE_SPECS]


def fit_shrinkage_priors(df):
    """Learn the shrinkage target (population mean of each raw rate column)
    from a training dataframe. Returns {rate_col: prior_mean}."""
    return {rate_col: float(df[rate_col].mean()) for rate_col, _ in SHRINKAGE_SPECS}


def add_shrinkage_columns(df, priors, k=50.0):
    """Beta-Binomial-style posterior-mean shrinkage: shrunk = (n*rate + k*prior)
    / (n+k). A pitcher with n=0 gets exactly the prior; n>>k gets ~raw rate.
    k is the "prior sample-size equivalent" -- how many observations of
    the prior it takes to counterbalance the raw rate.
    """
    df = df.copy()
    for rate_col, n_col in SHRINKAGE_SPECS:
        n = pd.to_numeric(df[n_col], errors="coerce").fillna(0.0)
        rate = pd.to_numeric(df[rate_col], errors="coerce").fillna(0.0)
        prior = priors[rate_col]
        df[f"shrunk_{rate_col}"] = (n * rate + k * prior) / (n + k)
    return df


# Pitcher-batter matchup history. Unlike every other feature in this file,
# this is NOT a pure row-local transform -- it requires the pair's own
# history, which for TRAINING rows must be computed leak-safe (only rows
# strictly BEFORE the current one, using train.csv's row order, which we
# verified is a valid chronological proxy -- see
# training/verify_row_order.py). For INFERENCE rows (test.csv), there is no
# "before" to expand over within the row itself; the pair's full known
# history is the static, precomputed lookup table built once from ALL of
# train.csv (2019-2024), entirely prior to the 2025 test season -- the same
# asof discipline the organizers used for the official asof_* columns.
MATCHUP_SHRUNK_COL = "matchup_shrunk_success_rate"
MATCHUP_N_COL = "matchup_n"


def _expanding_target_prior(df):
    """As-of (leak-safe) GLOBAL expanding mean of TARGET_COL, using only
    rows strictly BEFORE the current one in df's row order. Used as the
    shrinkage-anchor "prior_mean" everywhere below.

    Replaces a previous bug: both add_matchup_columns_training and
    add_matchup_extension_columns_training used to compute
    `float(df[TARGET_COL].mean())` -- a SINGLE scalar over the entire
    dataframe passed in. Since tuning_common.load_split() (and
    train_final_v2.py) call these on the full raw train.csv before
    splitting into TR/CAL/VAL, that scalar was computed using CAL/VAL
    rows' actual labels when building features for earlier TR rows -- a
    genuine look-ahead violation, caught by verify_extension_leakage.py's
    causality truncation test (quantified gap: full 2019-2024 mean
    0.5238 vs TR-only 2019-2022 mean 0.5395, a 0.0158 anchor shift).

    This expanding version is consistent with how cum_n/cum_success are
    already computed elsewhere in this file: strictly prior rows only, no
    future information, and it naturally adapts per-split with no
    hardcoded boundary. The very first row (no prior data at all) falls
    back to a fixed neutral constant (0.5), not derived from any row's own
    label -- affects exactly 1 row out of ~1.47M, immaterial.
    """
    target = df[TARGET_COL].to_numpy(dtype=float)
    n = len(target)
    cum_sum_before = np.concatenate(([0.0], np.cumsum(target)[:-1]))
    cum_count_before = np.arange(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        prior = np.where(cum_count_before > 0, cum_sum_before / cum_count_before, np.nan)
    prior = np.nan_to_num(prior, nan=0.5)
    return pd.Series(prior, index=df.index)


def add_matchup_columns_training(df, k=30.0):
    """Leak-safe expanding (pitcher_id, batter_id) success rate, computed
    from train.csv's own row order. Requires TARGET_COL to be present."""
    df = df.copy()
    grp = df.groupby(["pitcher_id", "batter_id"])[TARGET_COL]
    cum_n = grp.cumcount().astype(float)              # prior meetings (excludes current row)
    cum_success = grp.cumsum() - df[TARGET_COL]        # prior successes only
    prior_mean = _expanding_target_prior(df)
    raw_rate = np.where(cum_n > 0, cum_success / cum_n.replace(0, np.nan), prior_mean)
    df[MATCHUP_N_COL] = cum_n
    df[MATCHUP_SHRUNK_COL] = (cum_n * np.nan_to_num(raw_rate, nan=0.0) + k * prior_mean) / (cum_n + k)
    return df


def fit_matchup_lookup(df, k=30.0):
    """Build the static (pitcher_id, batter_id) -> (n, shrunk_rate) lookup
    table from the FULL training set, for use at inference time."""
    prior_mean = float(df[TARGET_COL].mean())
    g = df.groupby(["pitcher_id", "batter_id"])[TARGET_COL].agg(["count", "mean"])
    g["shrunk"] = (g["count"] * g["mean"] + k * prior_mean) / (g["count"] + k)
    lookup = {
        f"{pid}_{bid}": {"n": int(row["count"]), "rate": float(row["shrunk"])}
        for (pid, bid), row in g[["count", "shrunk"]].iterrows()
    }
    return {"prior_mean": prior_mean, "pairs": lookup}


def add_matchup_columns_inference(df, matchup_table):
    """Apply the static matchup lookup table to inference rows."""
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


# =======================================================================
# Round 3: matchup feature extensions. All three require
# add_matchup_columns_training to have already been run on df (needs
# MATCHUP_SHRUNK_COL present for the interaction terms).
# =======================================================================

MATCHUP_LAST_N = 3
MATCHUP_LASTN_COL = f"matchup_last{MATCHUP_LAST_N}_success_rate"
MATCHUP_HAND_COL = "pitcher_vs_hand_shrunk_success_rate"
MATCHUP_INTERACTION_PITCHER_COL = "matchup_x_pitcher_rate"
MATCHUP_INTERACTION_BATTER_COL = "matchup_x_batter_rate"
MATCHUP_EXTENSION_COLS = [
    MATCHUP_LASTN_COL, MATCHUP_HAND_COL,
    MATCHUP_INTERACTION_PITCHER_COL, MATCHUP_INTERACTION_BATTER_COL,
]


def add_matchup_extension_columns_training(df, k=30.0, last_n=MATCHUP_LAST_N):
    """Leak-safe, row-order-based (see add_matchup_columns_training):
    1) last-N-meetings rolling trend per (pitcher_id, batter_id) pair --
       does this specific matchup look different recently vs its full
       history? (shift(1) excludes the current row from its own window)
    2) pitcher's shrunk success rate specifically vs the CURRENT row's
       batter_hand -- a pitcher-specific platoon-split signal, distinct
       from both the pooled hand_matchup categorical and the exact-pair
       matchup rate (far more data per pitcher-hand group than per pair).
    3) matchup rate x each player's own asof rate (same interaction
       pattern as pitcher_batter_form elsewhere in this file).
    """
    df = df.copy()
    prior_mean = _expanding_target_prior(df)  # fixed: was a full-dataframe scalar (leak), see _expanding_target_prior's docstring

    df[MATCHUP_LASTN_COL] = df.groupby(["pitcher_id", "batter_id"])[TARGET_COL].transform(
        lambda s: s.shift(1).rolling(last_n, min_periods=1).mean()
    )
    df[MATCHUP_LASTN_COL] = df[MATCHUP_LASTN_COL].fillna(prior_mean)

    hand_grp = df.groupby(["pitcher_id", "batter_hand"])[TARGET_COL]
    cum_n = hand_grp.cumcount().astype(float)
    cum_success = hand_grp.cumsum() - df[TARGET_COL]
    raw_rate = np.where(cum_n > 0, cum_success / cum_n.replace(0, np.nan), prior_mean)
    df[MATCHUP_HAND_COL] = (cum_n * np.nan_to_num(raw_rate, nan=0.0) + k * prior_mean) / (cum_n + k)

    df[MATCHUP_INTERACTION_PITCHER_COL] = df[MATCHUP_SHRUNK_COL] * df["asof_pitcher_success_rate"].fillna(prior_mean)
    df[MATCHUP_INTERACTION_BATTER_COL] = df[MATCHUP_SHRUNK_COL] * df["asof_batter_success_rate"].fillna(prior_mean)
    return df


def fit_matchup_extension_lookup(df):
    """Static lookup tables for inference (df must already have
    add_matchup_extension_columns_training applied): each pair/pitcher-hand
    key maps to the value from its LAST row in train.csv -- the most
    complete state after all of 2019-2024, mirroring fit_matchup_lookup's
    design. Interactions need no lookup: computed directly at inference
    time from already-loaded features."""
    prior_mean = float(df[TARGET_COL].mean())

    last_pair_rows = df.drop_duplicates(subset=["pitcher_id", "batter_id"], keep="last")
    pair_lastn = {
        f"{pid}_{bid}": float(v) for pid, bid, v in
        zip(last_pair_rows.pitcher_id, last_pair_rows.batter_id, last_pair_rows[MATCHUP_LASTN_COL])
    }

    last_hand_rows = df.drop_duplicates(subset=["pitcher_id", "batter_hand"], keep="last")
    hand_lookup = {
        f"{pid}_{bh}": float(v) for pid, bh, v in
        zip(last_hand_rows.pitcher_id, last_hand_rows.batter_hand, last_hand_rows[MATCHUP_HAND_COL])
    }

    return {"prior_mean": prior_mean, "pair_lastn": pair_lastn, "pitcher_hand": hand_lookup}


def add_matchup_extension_columns_inference(df, ext_table):
    """Requires MATCHUP_SHRUNK_COL already present (add_matchup_columns_inference
    must run first)."""
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


# =======================================================================
# Round 3, item #8 (lowest priority): coarse trackman_history.csv
# season-level environmental priors. trackman_history.csv only covers
# 2019-2024 (no 2025), so test.csv (season 2025) rows use 2024's values --
# justified empirically: rel_speed/spin_rate are essentially flat across
# 2022-2024 (a real jump happened 2021->2022, but no further trend since),
# so "most recent known season" is a reasonable extrapolation, unlike
# control_success's steady multi-year decline that recency-weighting was
# built to handle. If this doesn't move validation BSS, it gets dropped --
# these are genuinely environmental/coarse, not asserted to be strong
# signal.
# =======================================================================

TRACKMAN_METRICS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension"]
TRACKMAN_PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
TRACKMAN_ENV_COLS = [f"env_{grp}_{metric}" for grp in TRACKMAN_PITCH_GROUPS for metric in TRACKMAN_METRICS]


def fit_trackman_env_table(trackman_path):
    """Season x pitch_type_group mean of each metric, pivoted wide to one
    row per season. Returns a small DataFrame (season, env_*_* columns)."""
    tk = pd.read_csv(trackman_path, usecols=["season", "pitch_type_group"] + TRACKMAN_METRICS)
    g = tk.groupby(["season", "pitch_type_group"])[TRACKMAN_METRICS].mean().reset_index()
    wide = g.pivot(index="season", columns="pitch_type_group", values=TRACKMAN_METRICS)
    wide.columns = [f"env_{grp}_{metric}" for metric, grp in wide.columns]
    wide = wide.reset_index()
    return wide


def add_trackman_env_columns(df, env_table):
    """Join by season; seasons beyond the table's max (i.e. 2025 test data)
    use the max known season's row (see module docstring above)."""
    df = df.copy()
    max_season = int(env_table["season"].max())
    join_season = df["season"].clip(upper=max_season).rename("season")
    merged = pd.DataFrame({"season": join_season}).merge(env_table, on="season", how="left")
    for col in TRACKMAN_ENV_COLS:
        df[col] = merged[col].values
    return df
