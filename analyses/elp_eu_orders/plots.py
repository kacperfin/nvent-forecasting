import pandas as pd
import matplotlib
import seaborn as sns

color_palette = sns.color_palette()

def plot_prediction(ax, target:str, y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> None:

    y_train[target].plot(ax=ax, label='Train data', color=color_palette[0], **kwargs)
    y_test[target].plot(ax=ax, label='Test data', color=color_palette[0], **kwargs)
    y_pred[target].plot(ax=ax, label='Prediction', color=color_palette[1], style='--', **kwargs)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.legend()

    ax.set_xlabel('Date')
    ax.set_ylabel('$', rotation=0, labelpad=16)