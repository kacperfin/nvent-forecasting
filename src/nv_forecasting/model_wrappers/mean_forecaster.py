import numpy as np

from .forecaster_base import ForecasterBase

class MeanForecaster(ForecasterBase):
    def __init__(self, last_days: int=None):
        self.name = 'mean_forecaster'
        self.mean_value = None
        self.last_days = last_days

    def fit(self, X, y, cv, scoring):
        if self.last_days is None:
            self.mean_value = np.mean(y)
        else:
            self.mean_value = np.mean(y.iloc[-self.last_days:])

    def predict(self, X):
        return np.array([self.mean_value] * len(X))
    
    def get_best_params(self):
        return {}