import subprocess
import itertools
import numpy as np
import pandas as pd


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


def csv_list_fields(csv_in=''):
    '''
    Returns field name of a csv file as a list of strings

    Parameters
    ------------
    csv_in : str
        Full path of input csv file

    Returns
    --------
    fields : list
        Field names as a list of strings
    '''
    return subprocess.check_output(
        'head -1 {}'.format(csv_in), shell=True
    ).decode('utf8').strip().split(',')


def mapper_label2count(df, col=''):
    '''
    Replacing high-cardinality categorical labels with normalized risk factor
    normalized risk factor = target mean of each label normalized by the
    grand target mean of all labels
    Mapping should be generated with training data, and then apply the
    resulting mapping to testing data (no testing data snooping!)

    Parameters
    ------------
    df : pandas.DataFrame
        Input data frame to be processed; prediction target must be included
    col : str
        Feature name to be processed

    Returns
    ----------
    mapper : dict
        Dictionary with mapping from labels to risk factor values

    Examples
    ---------
    Process 'ip' field of the input dataframe

    >>> mapper = preprocessing.mapper_label2riskfactor(df, col='ip')
    >>> df[col + '_risk'] = df[col].map(mapper)
    '''
    return df[col].value_counts().to_dict()


def mapper_label2riskfactor(df, col=''):
    '''
    Replacing high-cardinality categorical labels with normalized risk factor
    normalized risk factor = target mean of each label normalized by the
    grand target mean of all labels
    Mapping should be generated with training data, and then apply the
    resulting mapping to testing data (no testing data snooping!)

    Parameters
    ------------
    df : pandas.DataFrame
        Input data frame to be processed; prediction target must be included
    col : str
        Feature name to be processed

    Returns
    ----------
    mapper : dict
        Dictionary with mapping from labels to risk factor values

    Examples
    ---------
    Process 'ip' field of the input dataframe

    >>> mapper = preprocessing.mapper_label2riskfactor(df, col='ip')
    >>> df[col + '_risk'] = df[col].map(mapper)
    '''
    p_normalization = df['is_attributed'].mean()
    prob_df = pd.DataFrame(df.groupby([col])['is_attributed'].mean())
    prob_df['risk'] = prob_df.is_attributed / p_normalization
    return prob_df['risk'].to_dict()

def df_rarelabel_imputer(
    df, cols=[], thresh_percentage=1.0, replace_with=1e10
):
    '''
    Replace rare labels with dummy value. Critical for avoiding overfitting

    Parameters
    ------------
    df : pandas.DataFrame
        Input dataframe to be processed
    cols : list
        list of column names to be processed
    thresh_percentage : float
        Threshold for defining rare labels
    replace_with : float
        Dummy value used for busketing all rare labels to the same group
        Should be a value not present in any raw labels

    Returns
    ----------
    df : pandas.DataFrame
        Output dataframe whose rare labels have been imputed

    Examples
    ---------
    Replace values in ['ip', 'app', 'channel'] that are only present in less
    than 1% of the observations with a large number 1e10

    >>> df = impute_rare_label(
    ... df, cols=['ip', 'app', 'channel'], thresh_percentage=1.0,
    ... replace_with=1e10
    ... )
    '''
    # Turn off SettingWithCopyWarning from pandas
    pd.options.mode.chained_assignment = None
    for col in cols:
        # Compute proportion of all value counts in percentage
        df_temp = df[col].value_counts() / len(df) * 100.0
        labels_rare = df_temp[df_temp <= thresh_percentage].index
        if len(labels_rare) > 0:
            # Impute rare labels; keep other labels intact
            df[col] = np.where(
                df[col].isin(labels_rare), replace_with, df[col]
            )
    return df


def df_feature_grouper(df, cols=[], r=2):
    feature_groups = itertools.combinations(cols, r)
    for group in feature_groups:
        name = 'count_{}'.format('-'.join(group))
        df[name] = df.merge(
            df.groupby(list(group)).count()['is_attributed'].reset_index(),
            how='left', on=list(group)
        )['is_attributed_y']
    return df.fillna(0)
