import pandas as pd
import numpy as np

class MeanForecaster():
    def __init__(self):
        self.mean_value = None

    def fit(self, y: pd.Series) -> None:
        self.mean_value = np.mean(y)

    def predict(self, df: pd.DataFrame) -> np.array:
        return np.array([self.mean_value] * len(df))