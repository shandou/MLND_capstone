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


def mapper_category2woe(df, col=''):
    p_ref = df['is_attributed'].sum() / len(df)
    prob_df = pd.DataFrame(df.groupby([col])['is_attributed'].mean())
    prob_df['ratio'] = prob_df.is_attributed / p_ref
    return prob_df['ratio'].to_dict()

