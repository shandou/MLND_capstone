import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_corrmat(df):
    target_col = 'is_attributed'
    cols_corr = [x for x in df.columns if x != target_col]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8, 4))
    fig.tight_layout()
    for i, axi in enumerate(ax):
        corr_matrix = df[df['is_attributed'] == i][cols_corr].corr()
        mask = np.zeros_like(corr_matrix, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(
            corr_matrix, mask=mask, cmap='RdBu', vmin=-1, vmax=1, square=True,
            linewidths=.5, annot=True, fmt='.1f', ax=axi, cbar=False
        )
        axi.set(title='is_attributed = {}'.format(i))
    plt.show()


def plot_corrmat_simple(df, figsize=(8, 8)):
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
