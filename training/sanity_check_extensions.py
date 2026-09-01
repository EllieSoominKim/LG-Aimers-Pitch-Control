import sys
sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
import pandas as pd
import numpy as np
import features as feat

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv", nrows=100000)
raw = feat.add_matchup_columns_training(raw, k=30.0)
raw = feat.add_matchup_extension_columns_training(raw, k=30.0)

print("Extension columns summary:")
print(raw[feat.MATCHUP_EXTENSION_COLS].describe())

# spot check the last-N rolling trend for a repeated pair
pair_counts = raw.groupby(["pitcher_id","batter_id"]).size()
rep_pair = pair_counts[pair_counts >= 8].index[0]
pid, bid = rep_pair
sub = raw[(raw.pitcher_id==pid) & (raw.batter_id==bid)][
    ["row_id","control_success", feat.MATCHUP_N_COL, feat.MATCHUP_LASTN_COL]]
print(f"\nSpot-check pair ({pid},{bid}) -- last{feat.MATCHUP_LAST_N} should reflect only the prior {feat.MATCHUP_LAST_N} outcomes:")
print(sub.to_string(index=False))

# manual verification of row 4 (0-indexed within pair): last3 should be mean of rows 1,2,3 (0-indexed) success values
vals = sub["control_success"].values
for i in range(len(vals)):
    window = vals[max(0,i-3):i]
    expected = window.mean() if len(window) else None
    actual = sub[feat.MATCHUP_LASTN_COL].values[i]
    match = "OK" if expected is None or abs(expected-actual)<1e-9 else f"MISMATCH (expected {expected})"
    print(f"  row {i}: actual={actual:.4f}  {match if expected is not None else '(cold start, uses prior_mean)'}")

# lookup table + inference round trip
ext_table = feat.fit_matchup_extension_lookup(raw)
print(f"\nExt lookup: {len(ext_table['pair_lastn'])} pair entries, {len(ext_table['pitcher_hand'])} pitcher-hand entries")

matchup_table = feat.fit_matchup_lookup(raw, k=30.0)
inf_test = raw[(raw.pitcher_id==pid) & (raw.batter_id==bid)].tail(1).copy()
inf_result = feat.add_matchup_columns_inference(inf_test, matchup_table)
inf_result = feat.add_matchup_extension_columns_inference(inf_result, ext_table)
print("\nInference on last row of that pair:")
print(inf_result[["row_id"] + [feat.MATCHUP_N_COL, feat.MATCHUP_SHRUNK_COL] + feat.MATCHUP_EXTENSION_COLS].to_string(index=False))
print("(matchup_lastN should match the TRAINING row's own value, since inference uses the pair's final known state)")
print("training row's value:", sub[feat.MATCHUP_LASTN_COL].values[-1])

print("\nALL SANITY CHECKS COMPLETE")
