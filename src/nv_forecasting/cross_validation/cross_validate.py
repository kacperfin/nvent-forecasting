from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import numpy as np
import pandas as pd

from nv_forecasting.model_wrappers.forecaster_base import ForecasterBase

def cross_validate(models: list[ForecasterBase], outer_cv: TimeSeriesSplit, inner_cv: TimeSeriesSplit, X: pd.DataFrame, y: pd.DataFrame, target: str, scoring: str, additional_aggregations: list[str] = None) -> pd.DataFrame:
    results = []

    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for model in models:
            model.fit(X_train, y_train[target], inner_cv, scoring)

            y_pred = model.predict(X_test)

            y_pred_df = pd.DataFrame(data=y_pred,
                                  columns=[target],
                                  index=y_test.index)
            
            data = {
                'model': model.name,
                'outer_fold': outer_fold,
                'train_start': X_train.index[0],
                'train_end': X_train.index[-1],
                'test_start': X_test.index[0],
                'test_end': X_test.index[-1],
                'test_size': len(X_test),
            }

            best_params = {}
            model_best_params = model.get_best_params()

            for param in model_best_params:
                best_params[param + '__p'] = model_best_params[param]
            data.update(best_params)

            scores = {
                'rmse': root_mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'mape': mean_absolute_percentage_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
            }

            data.update(scores)

            if additional_aggregations is not None:
                for agg in additional_aggregations:
                    agg_upper = agg.upper()
                    y_pred_agg = y_pred_df.resample(agg_upper).mean()
                    y_test_agg = y_test.resample(agg_upper).mean()

                    scores_agg = {
                        'rmse_' + agg_upper: root_mean_squared_error(y_test_agg, y_pred_agg),
                        'mae_' + agg_upper: mean_absolute_error(y_test_agg, y_pred_agg),
                        'mape_' + agg_upper: mean_absolute_percentage_error(y_test_agg, y_pred_agg),
                        'r2_' + agg_upper: r2_score(y_test_agg, y_pred_agg)
                    }

                    data.update(scores_agg)
           
            data.update(
                {
                    'y_pred': y_pred,
                }
            )

            results.append(data)

    return pd.DataFrame(results)