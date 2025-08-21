TARGET = 'y' # Target column name
COLUMNS_TO_DROP = [TARGET, 'count_of_orders']

# Feature engineering
LAGS = [182]
DIFFS = [182]

# Model training
N_OUTER_SPLITS = 4
N_INNER_SPLITS = 3
N_FINAL_SPLITS = 5
TEST_SIZE = 182
SCORING = 'neg_mean_squared_error'