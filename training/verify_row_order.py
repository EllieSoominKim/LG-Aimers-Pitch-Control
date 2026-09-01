import pandas as pd
import numpy as np

train = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv",
                     usecols=["row_id", "pitcher_id", "batter_id", "season", "game_month", "asof_pitcher_n"])
train = train.reset_index(drop=True)

# For a fixed pitcher_id, does asof_pitcher_n strictly increase as row index increases?
# (each subsequent row for that pitcher should have seen exactly one more prior pitch)
sample_pitchers = train.pitcher_id.value_counts().head(20).index
violations = 0
checked = 0
for pid in sample_pitchers:
    sub = train[train.pitcher_id == pid].sort_index()
    diffs = sub.asof_pitcher_n.diff().dropna()
    checked += len(diffs)
    violations += (diffs < 0).sum()
    non_monotonic_or_notplus1 = (diffs != 1).sum()

print(f"Checked {checked} consecutive same-pitcher row pairs across {len(sample_pitchers)} pitchers")
print(f"Violations (asof_pitcher_n DECREASED going down row order): {violations}")

# Also check season/game_month is non-decreasing as row index increases (global chronological order)
season_diffs = train.season.diff().dropna()
print(f"\nRows where season DECREASED vs previous row: {(season_diffs < 0).sum()} / {len(train)}")

month_within_season = train.groupby("season")["game_month"].apply(lambda s: (s.diff().dropna() < 0).sum())
print("Rows where game_month decreased within same season block:")
print(month_within_season)
