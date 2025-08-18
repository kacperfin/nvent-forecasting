import numpy as np
import pandas as pd

def add_time_features(df: pd.DataFrame) -> None:
    df['is_weekend'] = df.index.day_of_week >= 5
    df['is_workday'] = ~df.is_weekend
    df['years_from_start'] = df.index.year - np.min(df.index.year)
    df['quarter'] = (df.index.month - 1) // 3 + 1
    df['quarter'] = df.quarter.astype('category')
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    df['day'] = df.index.day
    df['day_of_year'] = df.index.day_of_year
    df['week_of_year'] = df.index.isocalendar().week

def add_lags(df: pd.DataFrame, lags: list[int], columns: list[str]):
    for column in columns:
        for lag in lags:
            df[f'{column}_lag_{lag}'] = df[column].shift(lag)

def add_diffs(df: pd.DataFrame, diffs: list[int], columns: list[str]):
    for column in columns:
        for diff in diffs:
            df[f'{column}_diff_{diff}'] = df[column].diff(diff)