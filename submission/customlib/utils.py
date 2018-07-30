'''
utils module is for plotting and for packing results into data frames
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, confusion_matrix, precision_recall_curve
)


def plot_corrmat(df, figsize=(8, 8)):
    '''
    Plots Pearson's correlation coefficients for input dataframe
    Only lower portion of the matrix is displayed
    '''
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


def plot_scores_vs_regparams(
        regparams=None, scores_train=None, scores_test=None, figsize=(4, 4),
        xscale='log', xlabel='', **kwargs
):
    '''
    Plots metrics as a function of regularization parameters
    '''
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(
        regparams, scores_train, 'b.-', label='Traing score', lw=3, ms=10
    )
    ax1.plot(
        regparams, scores_test, 'r.-', label='Testing score', lw=3, ms=10
    )
    ax1.legend()
    ax1.set(ylim=(0.8, 1), xlabel=xlabel, ylabel='AUC score')
    ax2 = ax1.twinx()
    ax2.plot(
        regparams, scores_train - scores_test, 'g.-', lw=3, ms=10, alpha=0.8
    )
    ax2.set_ylabel('AUC[train] - AUC[test]', color='g')
    ax2.tick_params(colors='g')
    if xscale is 'log':
        plt.xscale('log')
    plt.show()
    return None


def plot_feature_importance(
        model, feature_list=[], model_name='', figsize=(5, 5), **kwargs
):
    '''
    Bar plots of feature importance ranking
    '''
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


def plot_confusion_matrix(y_test, y_pred, figsize=(), **kwargs):
    '''
    Plot confusion matrix and return matrix as dataframe
    '''
    cm = confusion_matrix(y_test, y_pred)
    labels = np.array([
        ['TN: {:d}'.format(cm[0][0]), 'FP: {:d}'.format(cm[0][1])],
        ['FN: {:d}'.format(cm[1][0]), 'TP: {:d}'.format(cm[1][1])]
    ])
    df_cm = pd.DataFrame(
        data=cm, columns=['predicted_0', 'predicted_1'],
        index=['actual_0', 'actual_1']
    )
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        df_cm, annot=labels, cmap='Blues', square=True, linewidths=.5,
        cbar=False, fmt=''

    )
    plt.show()
    return df_cm


def plot_roc_curve(
        y_test, y_pred_proba, figsize=(5, 5), **kwargs
):
    '''
    Plot ROC curve and returns value for further comparions/visualizations
    '''
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(fpr, tpr, 0, where=(tpr > 0), alpha=0.5, color='silver')
    ax.plot(fpr, tpr, **kwargs)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel='False positive rate',
        ylabel='True positive rate', title='ROC curve'
    )
    plt.show()
    return np.vstack((fpr, tpr))


def plot_precision_vs_recall(y_test, y_pred_proba, figsize=(5, 5), **kwargs):
    '''
    Plot precision-recall curve and return values for
    further comparisons/visualizations
    '''
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, y_pred_proba[:, 1]
    )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(recalls, precisions, **kwargs)
    ax.set(
        xlim=(0, 1), ylim=(0, 1), xlabel='Recall',
        ylabel='Precision', title='Precision vs. recall curve'
    )
    plt.show()
    return np.vstack((recalls, precisions))


def df_check_cardinality(df):
    '''
    Utility for checking cardinality for input dataframe
    '''
    df_counts = pd.DataFrame()
    df_counts['n_unique'] = df.nunique()
    df_counts['n_unique (%)'] = 100 * df_counts['n_unique'] / len(df)
    return df_counts.T


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
