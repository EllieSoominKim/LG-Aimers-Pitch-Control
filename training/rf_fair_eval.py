"""Fair (no-leakage) comparison: replicate baseline's RF pipeline but fit
ONLY on TR=2019-2022 and evaluate on VAL=2024, same split as our other
experiments. The original baseline_submit/model/rf.pkl was almost certainly
trained on the full 2019-2024 train.csv, so evaluating it "on 2024" is
in-sample for those rows -- not a fair number to compare against.
"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

sys.path.insert(0, "C:/LG-Aimers-Pitch-Control/training")
from metrics import brier_skill_score

raw = pd.read_csv("C:/LG-Aimers-Pitch-Control/data/train.csv")
tr = raw[raw["season"] <= 2022].drop(columns=["row_id"])
val = raw[raw["season"] == 2024].drop(columns=["row_id"])
y_tr, y_val = tr["control_success"].values.astype(float), val["control_success"].values.astype(float)
X_tr, X_val = tr.drop(columns=["control_success"]), val.drop(columns=["control_success"])
p_clim = y_tr.mean()

cat_cols = ["top_bottom", "game_type", "base_state"]
num_cols = [c for c in X_tr.columns if c not in cat_cols]

pipe = Pipeline([
    ("pre", ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])),
    ("clf", RandomForestClassifier(max_depth=10, min_samples_leaf=200, n_estimators=100, n_jobs=6, random_state=42)),
])

t0 = time.time()
pipe.fit(X_tr, y_tr)
p = pipe.predict_proba(X_val)[:, 1]
bss, bs, bs_ref = brier_skill_score(y_val, p, p_clim)
print(f"RF (baseline arch, fit on TR only) VAL: Brier={bs:.5f}  BSS={bss:.5f}  ({time.time()-t0:.1f}s)")
