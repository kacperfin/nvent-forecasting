import pandas as pd
from settings import TARGET

def get_elp_eu_orders_dataframe() -> pd.DataFrame:
    df = pd.read_csv('../../data/elp_eu_orders_daily.csv')
    df.columns = [column.lower() for column in df.columns]
    df['ds'] = pd.to_datetime(df['ds'])
    df.set_index('ds', inplace=True)
    df = df.resample('D').sum()

    # Manual adjustment - noticed an anomaly
    df.loc['2024-09-17', TARGET] = df.loc['2024-09-17', TARGET] + df.loc['2024-09-18', TARGET]
    df.loc['2024-09-18', TARGET] = 0

    return df