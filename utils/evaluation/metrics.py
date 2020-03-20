from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import numpy as np


def hit_rate(gt_items_idx, predicted_items_idx):
    return len(set(gt_items_idx).intersection(set(predicted_items_idx)))
