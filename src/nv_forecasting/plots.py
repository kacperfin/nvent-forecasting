import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.model_selection import TimeSeriesSplit

color_palette = sns.color_palette()

def plot_cross_validation_results(y: pd.DataFrame, target: str, outer_cv: TimeSeriesSplit, cv_summary: pd.DataFrame, models_names_list: list[str], figsize: tuple[int, int] = (20, 3), **kwargs) -> None:
    n_outer_splits = outer_cv.get_n_splits()
    fig, ax = plt.subplots(n_outer_splits, 1, figsize=(figsize[0], figsize[1]*n_outer_splits), sharex=True)

    for outer_fold, (train_and_val_idx, test_idx) in enumerate(outer_cv.split(y)):
        y_train = y.iloc[train_and_val_idx]
        y_test = y.iloc[test_idx]

        y_train[target].plot(ax=ax[outer_fold], color=color_palette[0], label='Train data', **kwargs)
        y_test[target].plot(ax=ax[outer_fold], color=color_palette[0], label='Test data', alpha=0.5, **kwargs)

        pred_color = color_palette[1]
        pred_style = '--'
        pred_alpha = 1

        for i, model_name in enumerate(models_names_list):
            if i == 1:
                pred_color = '#000'
                pred_style = '-'
                pred_alpha = 0.2

            mask = (cv_summary.model == model_name) & (cv_summary.outer_fold == outer_fold)
            y_pred = cv_summary.loc[mask, 'y_pred'].iloc[0]
            y_pred = pd.Series(data=y_pred, index=y_test.index)
            y_pred.plot(ax=ax[outer_fold], color=pred_color, label=model_name, style=pred_style, alpha=pred_alpha, **kwargs)

        _format_axis(ax[outer_fold])

def plot_train_val(ax: Axes, target: str, train: pd.DataFrame, val: pd.DataFrame, **kwargs) -> None:
    train[target].plot(ax=ax, color=color_palette[0], label='Train data', **kwargs)
    val[target].plot(ax=ax, color=color_palette[1], label='Validation data', **kwargs)

    _format_axis(ax)

def plot_train_val_test(ax: Axes, target: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, **kwargs) -> None:
    train[target].plot(ax=ax, color=color_palette[0], label='Train data', **kwargs)
    val[target].plot(ax=ax, color=color_palette[1], label='Validation data', **kwargs)
    test[target].plot(ax=ax, color=color_palette[4], label='Test data', **kwargs)

    _format_axis(ax)

def plot_data_split(target: str, cv: TimeSeriesSplit, df: pd.DataFrame, figsize: tuple[int, int] = (20, 3), **kwargs) -> None:
    n_splits = cv.get_n_splits()
    
    fig, ax = plt.subplots(n_splits, 1, figsize=(figsize[0], figsize[1]*n_splits), sharex=True)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
        train, val = df.iloc[train_idx], df.iloc[val_idx]

        plot_train_val(ax[fold], target, train, val, linewidth=0.5)

def plot_nested_data_split(target: str, outer_cv: TimeSeriesSplit, inner_cv: TimeSeriesSplit, df: pd.DataFrame, figsize: tuple[int, int] = (20, 3), **kwargs) -> None:
    n_outer_splits = outer_cv.get_n_splits()
    n_inner_splits = inner_cv.get_n_splits()

    fig, ax = plt.subplots(n_outer_splits*n_inner_splits, 1, figsize=(figsize[0], figsize[1]*n_outer_splits*n_inner_splits), sharex=True)

    for outer_fold, (train_and_val_idx, test_idx) in enumerate(outer_cv.split(df)):
        train_and_val, test = df.iloc[train_and_val_idx], df.iloc[test_idx]
        for inner_fold, (train_idx, val_idx) in enumerate(inner_cv.split(train_and_val)):
            train, val = df.iloc[train_idx], df.iloc[val_idx]
            ax_idx = outer_fold * n_inner_splits + inner_fold
            plot_train_val_test(ax[ax_idx], target, train, val, test, **kwargs)

def plot_prediction(ax: Axes, target: str, y_train: pd.DataFrame, y_test: pd.DataFrame, y_pred: pd.DataFrame, **kwargs) -> None:
    y_train[target].plot(ax=ax, label='Train data', color=color_palette[0], **kwargs)
    y_test[target].plot(ax=ax, label='Test data', color=color_palette[4], **kwargs)
    y_pred[target].plot(ax=ax, label='Prediction', color=color_palette[1], style='--', **kwargs)

    _format_axis(ax)

def _format_axis(ax: Axes) -> None:
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Date')
    ax.set_ylabel('$', rotation=0, labelpad=16)
    ax.legend()