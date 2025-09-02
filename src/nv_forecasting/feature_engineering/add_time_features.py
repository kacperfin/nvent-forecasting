import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> None:
    df['is_weekend'] = df.index.day_of_week >= 5
    df['years_from_start'] = df.index.year - np.min(df.index.year)
    df['quarter'] = (df.index.month - 1) // 3 + 1
    df['quarter'] = df.quarter.astype('category')
    df['month_sin'] = np.sin(2 * np.pi * (df.index.month - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df.index.month - 1) / 12)
    df['day'] = df.index.day
    df['day_of_year'] = df.index.day_of_year
    df['week_of_year'] = df.index.isocalendar().week