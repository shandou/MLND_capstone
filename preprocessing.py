import subprocess
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
    df_train, df_test, cols=[], thresh_percentage=1.0, replace_with=1e10
):
    '''
    Replace rare labels with dummy value. Critical for avoiding overfitting

    Parameters
    ------------
    df_train : pandas.DataFrame
        Input training dataframe to be processed
    df_test : pandas.DataFrame
        Input testing dataframe to be processed
    cols : list
        list of column names to be processed
    thresh_percentage : float
        Threshold for defining rare labels
    replace_with : float
        Dummy value used for busketing all rare labels to the same group
        Should be a value not present in any raw labels

    Returns
    ----------
    df_train : pandas.DataFrame
        Output training dataframe whose rare labels have been imputed
    df_test : pandas.DataFrame
        Output testing dataframe whose rare labels have been imputed

    Examples
    ---------
    Replace values in ['ip', 'app', 'channel'] that are only present in less
    than 1% of the observations with a large number 1e10

    >>> df_train, df_test = impute_rare_label(
    ... df_train, df_test, cols=['ip', 'app', 'channel'],
    ... thresh_percentage=1.0, replace_with=1e10
    ... )
    '''
    # Turn off SettingWithCopyWarning from pandas
    pd.options.mode.chained_assignment = None
    for col in cols:
        # Compute proportion of all value counts in percentage
        df_temp = df_train[col].value_counts() / len(df_train) * 100.0
        labels_rare = df_temp[df_temp <= thresh_percentage].index
        if len(labels_rare) > 0:
            # Impute rare labels; keep other labels intact
            df_train[col] = np.where(
                df_train[col].isin(labels_rare), replace_with, df_train[col]
            )
            df_test[col] = np.where(
                df_test[col].isin(labels_rare), replace_with, df_test[col]
            )
    return (df_train.astype(int), df_test.astype(int))


def df_label2num_encoding(df_train, df_test, cols=[]):
    # Turn off SettingWithCopyWarning from pandas
    pd.options.mode.chained_assignment = None
    for col in cols:
        # risk factor features
        mapper_risk = mapper_label2riskfactor(df_train, col=col)
        df_train['risk_{}'.format(col)] = df_train[col].map(mapper_risk)
        df_test['risk_{}'.format(col)] = df_test[col].map(mapper_risk)
        # count features
        mapper_count = mapper_label2count(df_train, col=col)
        df_train['count_{}'.format(col)] = df_train[col].map(mapper_count)
        df_test['count_{}'.format(col)] = df_test[col].map(mapper_count)
        # Remove raw categorical features
        df_train.drop(columns=[col], inplace=True)
        df_test.drop(columns=[col], inplace=True)
    return (df_train, df_test)

