import pandas as pd
import numpy as np

train = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv",
                     usecols=["pitcher_id", "batter_id", "season", "control_success"])
print("Total rows:", len(train))

pair_counts = train.groupby(["pitcher_id", "batter_id"]).size()
print("\nUnique (pitcher_id, batter_id) pairs:", len(pair_counts))
print("Total rows covered by pairs with n>=1 (all rows):", pair_counts.sum())
print()
print("Distribution of pair sizes:")
print(pair_counts.describe())
print()
for thresh in [1, 2, 3, 5, 10, 20, 50]:
    n_pairs = (pair_counts >= thresh).sum()
    n_rows = pair_counts[pair_counts >= thresh].sum()
    pct_rows = n_rows / len(train) * 100
    print(f"  pairs with >= {thresh:3d} prior meetings: {n_pairs:6d} pairs, covering {n_rows:8d} rows ({pct_rows:.2f}% of train)")

print()
print("Rows where this exact pair has been seen at least once BEFORE (i.e. n>=2 total incl. current) vs completely novel pair (n==1, cold start):")
n1 = (pair_counts == 1).sum()
print(f"  pairs appearing exactly once (no repeat at all): {n1} ({n1/len(pair_counts)*100:.1f}% of pairs)")

# Now check: for each row, does the SAME (pitcher,batter) pair appear in an EARLIER row (asof sense)?
# Approximate via cumulative count check per pair ordered by original row order (proxy for time order)
train_sorted = train.reset_index(drop=True)
train_sorted["pair"] = list(zip(train_sorted.pitcher_id, train_sorted.batter_id))
train_sorted["pair_cumcount"] = train_sorted.groupby("pair").cumcount()
pct_with_prior_meeting = (train_sorted["pair_cumcount"] > 0).mean() * 100
print(f"\nRows where this exact (pitcher,batter) pair has appeared at least once earlier in the data: {pct_with_prior_meeting:.2f}%")

# breakdown by season
print()
print("Same stat by season:")
print(train_sorted.groupby("season")["pair_cumcount"].apply(lambda s: (s>0).mean()*100).round(2))
