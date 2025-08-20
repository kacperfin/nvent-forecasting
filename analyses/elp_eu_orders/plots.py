import pandas as pd
import matplotlib
import seaborn as sns

color_palette = sns.color_palette()

def plot_data_split(ax, target: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train[target].plot(ax=ax, color=color_palette[0], label='Train data', linewidth=0.5)
    val[target].plot(ax=ax, color=color_palette[1], label='Validation data', linewidth=0.5)
    test[target].plot(ax=ax, color=color_palette[4], label='Test data', linewidth=0.5)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Date')
    ax.set_ylabel('$', rotation=0, labelpad=16)
    ax.legend()

def plot_prediction(ax, target: str, y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> None:

    y_train[target].plot(ax=ax, label='Train data', color=color_palette[0], **kwargs)
    y_test[target].plot(ax=ax, label='Test data', color=color_palette[4], **kwargs)
    y_pred[target].plot(ax=ax, label='Prediction', color=color_palette[1], style='--', **kwargs)

    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Date')
    ax.set_ylabel('$', rotation=0, labelpad=16)
    ax.legend()