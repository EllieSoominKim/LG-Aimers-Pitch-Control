"""Brier score / Brier Skill Score helpers."""

import numpy as np


def brier_score(y_true, p_pred, sample_weight=None):
    y_true = np.asarray(y_true, dtype=np.float64)
    p_pred = np.asarray(p_pred, dtype=np.float64)
    sq_err = (p_pred - y_true) ** 2
    if sample_weight is None:
        return float(sq_err.mean())
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    return float(np.average(sq_err, weights=sample_weight))


def brier_skill_score(y_true, p_pred, p_climatology, sample_weight=None):
    """BSS = 1 - BS(model) / BS(climatology reference forecast).

    p_climatology: scalar or array, the "no-skill" reference forecast
    (typically the base rate observed in a period that does not include
    the evaluation rows themselves).
    """
    bs_model = brier_score(y_true, p_pred, sample_weight)
    bs_ref = brier_score(y_true, np.full_like(np.asarray(y_true, dtype=np.float64), p_climatology), sample_weight)
    return 1.0 - bs_model / bs_ref, bs_model, bs_ref
