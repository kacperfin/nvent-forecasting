TARGET = 'y' # Target column name
COLUMNS_TO_DROP = [TARGET, 'count_of_orders']

# Feature engineering
LAGS = [182]
DIFFS = [182]

# Model training
N_OUTER_SPLITS = 5
N_INNER_SPLITS = 2
TEST_SIZE = 182