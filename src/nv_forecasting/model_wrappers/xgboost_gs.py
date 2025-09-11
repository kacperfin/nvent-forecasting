from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

from .forecaster_base import ForecasterBase

class XGBoostGS(ForecasterBase):
    def __init__(self, param_grid):
        self.name = 'xgboost_gs'
        self._param_grid = param_grid

    def fit(self, X, y, cv, scoring):
        self._model = GridSearchCV(
            estimator=XGBRegressor(
                random_state=42,
                n_jobs=-1,
                objective='reg:squarederror',
                enable_categorical=True
            ),
            param_grid=self._param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=3,
            error_score='raise',
        )
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)
    
    def get_best_params(self):
        return self._model.best_params_