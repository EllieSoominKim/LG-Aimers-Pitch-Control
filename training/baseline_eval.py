import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from metrics import brier_skill_score

t0 = time.time()
raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
val = raw[raw["season"] == 2024].drop(columns=["row_id"])
y_val = val["control_success"].values.astype(float)
X_val = val.drop(columns=["control_success"])

tr = raw[raw["season"] <= 2022]
p_clim = tr["control_success"].mean()

model = joblib.load("c:/LG-Aimers-Pitch-Control/baseline_submit/model/rf.pkl")
p = model.predict_proba(X_val)[:, 1]
bss, bs, bs_ref = brier_skill_score(y_val, p, p_clim)
print(f"Baseline RF on our 2024 holdout: Brier={bs:.5f}  BSS={bss:.5f}  (ref BS={bs_ref:.5f}, p_clim={p_clim:.4f})")
print(f"({time.time()-t0:.1f}s)")
