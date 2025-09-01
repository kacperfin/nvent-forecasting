import pandas as pd

def add_lags(df: pd.DataFrame, lags: list[int], columns: list[str]) -> None:
    for column in columns:
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)