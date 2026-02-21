import numpy as np

def apply_risk_overlay(proba, max_weight_per_stock=0.1):
    w = (proba - 0.5).clip(-max_weight_per_stock, max_weight_per_stock)
    gross = np.abs(w).sum()
    return w / gross if gross > 0 else w
