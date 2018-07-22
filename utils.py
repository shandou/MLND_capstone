import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corrmat(df, figsize=(8, 8)):
    target_col = 'is_attributed'
    cols_corr = [x for x in df.columns if x != target_col]
    fig, ax = plt.subplots(figsize=figsize)
    corr_matrix = df[cols_corr].corr()
    mask = np.zeros_like(corr_matrix, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        corr_matrix, mask=mask, cmap='RdBu', vmin=-1, vmax=1, square=True,
        linewidths=.5, annot=True, fmt='.1f', ax=ax, cbar=False
    )
    plt.show()
    return None


def plot_feature_importance(
    model, feature_list=[], model_name='', figsize=(5, 5), **kwargs
):
    features = np.array(feature_list).astype(str)
    index_sorted = np.argsort(abs(model.feature_importances))
    importance_sorted = model.feature_importances[index_sorted]
    features_sorted = features[index_sorted]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(
        np.arange(index_sorted.size), abs(importance_sorted), **kwargs
    )
    yticklabels = (
        ['{}.{}'.format(i, x) for i, x in zip(
            np.arange(1, index_sorted.size + 1)[::-1], features_sorted
        )]
    )
    ax.set(
        yticks=np.arange(index_sorted.size), yticklabels=yticklabels,
        xlabel='Feature importance', title=model_name
    )
    plt.show()
    return (features_sorted, importance_sorted)


def ml_performance_summary(train_score, test_score, model_name=''):
    '''
    Pack machine learning performance scores into dataframe
    '''
    pd.options.display.float_format = '{:.3f}'.format
    df = pd.DataFrame(
        columns=['model', 'auc_train', 'auc_cv_std', 'auc_test'],
        data=np.full((1, 4), np.nan)
    )
    df['model'] = model_name
    if isinstance(train_score, dict):
        df['auc_train'] = train_score['cv_mean']
        df['auc_cv_std'] = train_score['cv_std']
    else:
        df['auc_train'] = train_score
    df['auc_test'] = test_score
    return df


