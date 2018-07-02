import subprocess
import itertools


def csv_randomized_downsamp(csv_in='', csv_out='', fraction=0.01):
    '''
    Returns reduced csv file from raw input `csv_in`

    Parameters
    -----------
    csv_in : str
        Full path of input csv file to be downsampled
    csv_out : str
        Full path of output csv file stores downsampled data
    fraction : float
        Fraction of raw data to keep in `csv_out`

    Returns
    --------
    nlines_in : int
        Number of lines in input csv file
    nlines_out : int
        Number of lines in output csv file

    Examples
    ---------
    Extract 1% of samples

    >>> import preprocessing
    >>> nlines_in, nlines_out = preprocessing.randomized_downsamp(
            csv_in='./data/train.csv', csv_out='./data/train_sample.csv',
            fraction=0.01
        )
    '''
    result = subprocess.check_output('wc -l {}'.format(csv_in), shell=True)
    nlines_in = int(result.split()[0])
    nlines_out = int(fraction * nlines_in)
    subprocess.call('head -1 {} > {}'.format(csv_in, csv_out), shell=True)
    subprocess.call(
        'gshuf -n {} {} >> {}'.format(nlines_out, csv_in, csv_out),
        shell=True
    )
    return (nlines_in, nlines_out)


def list_feature_combinations(feature_primary='', feature_other=[]):
    '''
    List all feature combinations of interests
    '''
    feature_combinations = []
    for i in range(1, len(feature_other) + 1):
        # Generate all possible combinations from feature subsets
        combinations = itertools.combinations(feature_other, i)
        for combo in combinations:
            # Tie primary feature to non-primary feature combinations
            feature_combinations.append([feature_primary] + list(combo))
    return feature_combinations


def df_engineered(
    df_in=None, feature_combinations=[], col_ref='is_attributed', csv_out=''
):
    '''
    Generate dataframe with engineered features

    Parameters
    -----------
    df_in : pd.DataFrame
        Input dataframe with raw features
    feature_combinations : list
        List of feature combinations
    col_ref : str
        Reference column
    csv_out : str
        Full path of csv file to store dataframe with engineered features

    Returns
    --------
    df_out : pd.DataFrame
        Output dataframe with engineered features
    '''
    df_out = df_in.copy()
    for combo in feature_combinations:
        feature = 'count_{}'.format('_'.join(combo))
        df_temp = df_in.groupby(combo).count()[col_ref].reset_index()
        df_temp = df_temp.rename(columns={col_ref: feature})
        df_out[feature] = df_in.merge(df_temp, how='outer', on=combo)[feature]
    if csv_out:
        df_out.to_csv(csv_out, index=False)
    return df_out
