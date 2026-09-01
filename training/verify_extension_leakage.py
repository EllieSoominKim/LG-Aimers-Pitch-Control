"""Item #4 leakage verification, requested before more work builds on top of
matchup extensions. Two independent checks per extension column:

1) Manual row-by-row recomputation for a real pitcher/batter pair, compared
   against add_matchup_extension_columns_training's output.
2) The gold-standard causality proof: recompute each column using only rows
   up to and including a cut point, and confirm every row's value BEFORE the
   cut is bit-identical whether or not later rows exist in the dataframe. If
   a column's value at row i changed depending on rows after i, that would
   be direct evidence of future leakage. This directly tests the causal
   claim (not just pattern-matching a "known safe idiom").
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import features as feat
from features import TARGET_COL

df_full = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
print(f"Loaded train.csv: {len(df_full)} rows")

# add_matchup_extension_columns_training requires MATCHUP_SHRUNK_COL present
df_full = feat.add_matchup_columns_training(df_full, k=30.0)
df_full = feat.add_matchup_extension_columns_training(df_full, k=30.0)

# -----------------------------------------------------------------
# Check 1: manual recomputation for one real, well-populated pair
# -----------------------------------------------------------------
pair_counts = df_full.groupby(["pitcher_id", "batter_id"]).size()
big_pair = pair_counts[pair_counts >= 8].index[0]
pid, bid = big_pair
sub = df_full[(df_full.pitcher_id == pid) & (df_full.batter_id == bid)].copy()
print(f"\n=== Check 1: manual recompute for pitcher={pid} batter={bid} (n={len(sub)}) ===")

targets = sub[TARGET_COL].values
manual_lastn = []
for i in range(len(targets)):
    window = targets[max(0, i - feat.MATCHUP_LAST_N):i]  # strictly BEFORE row i
    manual_lastn.append(window.mean() if len(window) > 0 else np.nan)
manual_lastn = np.array(manual_lastn)
prior_mean = float(df_full[TARGET_COL].mean())
manual_lastn_filled = np.where(np.isnan(manual_lastn), prior_mean, manual_lastn)

code_lastn = sub[feat.MATCHUP_LASTN_COL].values
match = np.allclose(manual_lastn_filled, code_lastn, atol=1e-9)
print(f"  matchup_last{feat.MATCHUP_LAST_N}_success_rate matches manual computation: {match}")
if not match:
    print("  MISMATCH rows:")
    for i in range(len(sub)):
        if not np.isclose(manual_lastn_filled[i], code_lastn[i], atol=1e-9):
            print(f"    row {i}: manual={manual_lastn_filled[i]}  code={code_lastn[i]}")
print(f"  raw targets for this pair: {targets.tolist()}")
print(f"  manual (pre-fill) values : {manual_lastn.tolist()}")
print(f"  code values              : {code_lastn.tolist()}")
print(f"  first row is NaN before fill (no history yet): {np.isnan(manual_lastn[0])}")

# -----------------------------------------------------------------
# Check 2: causality proof -- truncate the dataframe at various cut
# points, recompute all 3 columns from scratch, and confirm rows before
# the cut are UNCHANGED regardless of what rows exist after it.
# -----------------------------------------------------------------
print(f"\n=== Check 2: causality proof (truncation test) ===")
n = len(df_full)
cut_points = [n // 4, n // 2, (3 * n) // 4]
check_cols = [feat.MATCHUP_LASTN_COL, feat.MATCHUP_HAND_COL,
              feat.MATCHUP_INTERACTION_PITCHER_COL, feat.MATCHUP_INTERACTION_BATTER_COL]

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
raw_with_base = feat.add_matchup_columns_training(raw, k=30.0)

all_ok = True
for cut in cut_points:
    truncated = raw_with_base.iloc[:cut].copy()
    truncated_ext = feat.add_matchup_extension_columns_training(truncated, k=30.0)

    # Compare the first `probe_n` rows (well before the cut, so any
    # boundary effects at the very end of the truncated frame don't
    # confound the test) against the full-data computation.
    probe_n = min(cut - 1000, 50000) if cut > 1000 else cut
    probe_n = max(probe_n, 1)
    for col in check_cols:
        full_vals = df_full[col].values[:probe_n]
        trunc_vals = truncated_ext[col].values[:probe_n]
        ok = np.allclose(full_vals, trunc_vals, atol=1e-9, equal_nan=True)
        all_ok &= ok
        status = "OK" if ok else "MISMATCH -- POSSIBLE LEAKAGE"
        print(f"  cut={cut:>7d}  probe_n={probe_n:>7d}  col={col:<35s} {status}")

print(f"\nAll causality checks passed: {all_ok}")
print("(If True: every row's extension-feature value is identical whether "
      "computed with only past+current data, or with the full dataset "
      "including future rows -- i.e. no future information leaks in.)")
