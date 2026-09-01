import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from metrics import brier_skill_score

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv", usecols=["season", "control_success"])
season = raw["season"].values
y = raw["control_success"].values.astype(float)
tr_mask, val_mask = season <= 2022, season == 2024
y_tr, season_tr, y_val = y[tr_mask], season[tr_mask], y[val_mask]

p_clim = y_tr.mean()
decay = 0.005
w = decay ** (2024 - season_tr)
p_weighted_const = np.average(y_tr, weights=w)
print(f"unweighted TR mean (p_clim) = {p_clim:.4f}")
print(f"recency-weighted TR mean (decay={decay}) = {p_weighted_const:.4f}")
print(f"actual VAL(2024) mean = {y_val.mean():.4f}")

bss, bs, bs_ref = brier_skill_score(y_val, np.full_like(y_val, p_weighted_const), p_clim)
print(f"\nConstant recency-weighted-mean prediction on VAL: Brier={bs:.5f}  BSS={bss:.5f}")
