import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from abc import ABC, abstractmethod

class ForecasterBase(ABC):
    name: str

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, cv: TimeSeriesSplit, scoring: str) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_best_params(self) -> dict:
        pass