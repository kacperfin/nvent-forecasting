from .data_handler import DataHandler

class ELPEUOrdersDaily(DataHandler):
    def _do_manual_adjustments(self, df):
        df.loc['2024-09-17', 'y'] = df.loc['2024-09-17', 'y'] + df.loc['2024-09-18', 'y']
        df.loc['2024-09-18', 'y'] = 0

        return df