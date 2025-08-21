import pandas as pd
import matplotlib
import seaborn as sns

color_palette = sns.color_palette()

def format_axis(ax):
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Date')
    ax.set_ylabel('$', rotation=0, labelpad=16)
    ax.legend() 

def plot_train_val(ax, target: str, train: pd.DataFrame, val: pd.DataFrame, **kwargs) -> None:
    train[target].plot(ax=ax, color=color_palette[0], label='Train data', **kwargs)
    val[target].plot(ax=ax, color=color_palette[1], label='Validation data', **kwargs)

    format_axis(ax)

def plot_train_val_test(ax, target: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, **kwargs) -> None:
    test[target].plot(ax=ax, color=color_palette[4], label='Test data', **kwargs)
    plot_train_val(ax, target, train, val, **kwargs)

    format_axis(ax)

def plot_prediction(ax, target: str, y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> None:

    y_train[target].plot(ax=ax, label='Train data', color=color_palette[0], **kwargs)
    y_test[target].plot(ax=ax, label='Test data', color=color_palette[4], **kwargs)
    y_pred[target].plot(ax=ax, label='Prediction', color=color_palette[1], style='--', **kwargs)

    format_axis(ax)