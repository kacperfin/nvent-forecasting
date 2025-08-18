import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from typing import NamedTuple

class Scores(NamedTuple):
    rmse: float
    mae: float
    mape: float
    r2: float

def get_scores(y_test: np.ndarray, y_pred: np.ndarray) -> Scores:
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return Scores(rmse, mae, mape, r2)

def add_scores_to_dict(metrics: dict, scores: Scores) -> None:
    if 'rmse' in metrics:
        metrics['rmse'].append(scores.rmse)

    if 'mae' in metrics:
        metrics['mae'].append(scores.mae)
    
    if 'mape' in metrics:
        metrics['mape'].append(scores.mape)
    
    if 'r2' in metrics:
        metrics['r2'].append(scores.r2)