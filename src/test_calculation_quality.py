import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
import os

def concatenate_df(path):
    '''
    Concatenates DataFrames from multiple files in a directory.

    Parameters:
    -----------
    path : str
        Directory path containing CSV files.

    Returns:
    --------
    df_out : DataFrame
        Concatenated DataFrame with columns 'id', 'leader', 'Ip', and 'Ie'.
    '''
    df_out = pd.DataFrame({'id': [], 'leader': [], 'Ip': [], 'Ie': []})
    for k in os.listdir(path):
        df = pd.read_csv(os.path.join(path, k))
        df_out = pd.concat([df_out, df])
    df_out['Ip'] = np.abs(df_out['Ip'])
    df_out['Ie'] = np.abs(df_out['Ie'])
    df_out.dropna(inplace=True)
    df_out.reset_index(inplace=True)
    return df_out

def test_difference(path_glob, path_loc):
    '''
    Performs a statistical test to compare the differences between two sets of data.

    Parameters:
    -----------
    path_glob : str
        Directory path containing global data CSV files.
    path_loc : str
        Directory path containing local data CSV files.

    Returns:
    --------
    statIp : float
        T-statistic for the comparison of Ip values.
    pIp : float
        Two-tailed p-value for the comparison of Ip values.
    statIe : float
        T-statistic for the comparison of Ie values.
    pIe : float
        Two-tailed p-value for the comparison of Ie values.
    '''
    df_glob = concatenate_df(path_glob)
    df_loc = concatenate_df(path_loc)
    IP_glob = df_glob['Ip']
    IP_loc = df_loc['Ip']
    statIp, pIp = ttest_ind(IP_glob, IP_loc, equal_var=False)
    IE_glob = df_glob['Ie']
    IE_loc = df_loc['Ie']
    statIe, pIe = ttest_ind(IE_glob, IE_loc, equal_var=False)
    return statIp, pIp, statIe, pIe

