import sys
sys.path.insert(0, r"C:\LG-Aimers-Pitch-Control")
import importlib.util
spec = importlib.util.spec_from_file_location("script", r"C:\LG-Aimers-Pitch-Control\script.py")
script = importlib.util.module_from_spec(spec)
# avoid running main() on import
import types
spec.loader.exec_module(script)

import pandas as pd, numpy as np, json

test = script.load_test(r"C:\LG-Aimers-Pitch-Control\data\test.csv")
categories = script.load_categories_with_fallback(r"C:\LG-Aimers-Pitch-Control\model")
calibrators = script.load_calibrators_with_fallback(r"C:\LG-Aimers-Pitch-Control\model")

X = script.build_features(test, categories)

lgbm = script._try_load(script._resolve_artifact_path(r"C:\LG-Aimers-Pitch-Control\model", "lightgbm.txt"), "lgb", lambda p: __import__("lightgbm").Booster(model_file=p))
from catboost import CatBoostClassifier
cb = CatBoostClassifier(); cb.load_model(r"C:\LG-Aimers-Pitch-Control\model\catboost.cbm")
import joblib
hgb = joblib.load(r"C:\LG-Aimers-Pitch-Control\model\hgb.joblib")

p_lgbm_raw = lgbm.predict(X)
p_cb_raw = cb.predict_proba(script.catboost_frame(X))[:, 1]
p_hgb_raw = hgb.predict_proba(script.hgb_frame(X))[:, 1]

p_lgbm_cal = calibrators["lightgbm"](p_lgbm_raw)
p_cb_cal = calibrators["catboost"](p_cb_raw)
p_hgb_cal = calibrators["hgb"](p_hgb_raw)

final = np.clip((p_lgbm_cal + p_cb_cal + p_hgb_cal) / 3.0, 0, 1)

cold_cols = ["row_id","asof_pitcher_n","asof_batter_n","asof_pitcher_success_rate","asof_batter_success_rate"]
diag = test[cold_cols].copy()
diag["lgbm_raw"] = p_lgbm_raw
diag["lgbm_cal"] = p_lgbm_cal
diag["cb_raw"] = p_cb_raw
diag["cb_cal"] = p_cb_cal
diag["hgb_raw"] = p_hgb_raw
diag["hgb_cal"] = p_hgb_cal
diag["final"] = final

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(diag.to_string(index=False))

print()
print("=== NaN pattern check for the two coldest rows ===")
for rid in ["TEST_005332", "TEST_035185"]:
    row = test[test.row_id == rid].iloc[0]
    n_nan_features = X[test.row_id.values == rid].isna().sum(axis=1).values[0]
    print(f"{rid}: asof_pitcher_n={row.asof_pitcher_n}, asof_batter_n={row.asof_batter_n}, "
          f"NaN features in model input = {n_nan_features} / {X.shape[1]}")
