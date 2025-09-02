from sklearn.model_selection import TimeSeriesSplit

DEFAULT_CONFIG = {
    'target': 'y', # Target column name
    'datetime_column_name': 'ds',
    'aggregation': 'D',
    'columns_to_lag': ['y', 'count_of_orders'],
    'columns_to_drop': ['y', 'count_of_orders'],

    # Feature engineering
    'lags': [182],

    # Model training
    'n_outer_splits': 4,
    'n_inner_splits': 3,
    'n_final_splits': 5,
    'test_size': 182,
    'scoring': 'neg_mean_squared_error'
}

# Data splits
DEFAULT_CONFIG['outer_cv'] = TimeSeriesSplit(n_splits=DEFAULT_CONFIG['n_outer_splits'], test_size=DEFAULT_CONFIG['test_size'])
DEFAULT_CONFIG['inner_cv'] = TimeSeriesSplit(n_splits=DEFAULT_CONFIG['n_inner_splits'], test_size=DEFAULT_CONFIG['test_size'])
DEFAULT_CONFIG['final_cv'] = TimeSeriesSplit(n_splits=DEFAULT_CONFIG['n_final_splits'], test_size=DEFAULT_CONFIG['test_size'])