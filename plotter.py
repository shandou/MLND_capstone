def compare_hist(df, by='', hist_params={}, ax=None):
    no_params = dict(facecolor='lightcoral', label='No download')
    yes_params = dict(facecolor='green', label='Yes download')
    ax.hist(
        df[df['is_attributed'] == 0][by], **hist_params, **no_params
    )
    ax.hist(
        df[df['is_attributed'] == 1][by], **hist_params, **yes_params
    )
    ax.legend(loc=4)
    ax.set(yscale='log', ylabel='Count')
    return ax


