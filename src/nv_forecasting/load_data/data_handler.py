import pandas as pd

class DataHandler():
    def __init__(self, file_path: str):
        self.file_path = file_path

    def get_dataframe(self, agg: str=None) -> pd.DataFrame:
        df = pd.read_csv(self.file_path)
        df = self._prepare_dataframe(df)
        
        if hasattr(self, '_do_manual_adjustments'):
            df = self._do_manual_adjustments(df)

        if agg:
            df = df.resample(agg).sum()

        return df
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [column.lower() for column in df.columns]
        df['ds'] = pd.to_datetime(df['ds'])
        df.set_index('ds', inplace=True)

        return df