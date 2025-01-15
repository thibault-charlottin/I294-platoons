from scipy.stats import f_oneway
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import kruskal

import pandas as pd
import os

def test_variability(df_list, Gipps=False):
    '''
    Performs a statistical test to compare the variability of multiple datasets.

    Parameters:
    -----------
    df_list : list
        List of DataFrames containing 'Tau' values.

    Returns:
    --------
    statistic : float
        F-statistic from the one-way ANOVA test.
    p_value : float
        Two-tailed p-value from the one-way ANOVA test.
    num_values : int
        Number of values used in the test.
    '''
    for d in range(len(df_list)):
        if Gipps == True:
            df_list[d]['Tau']=df_list[d]['reaction time']
        df_list[d].dropna(subset=['Tau'], inplace=True)
        grouped_df = df_list[d].groupby('id')['Tau'].agg(list).reset_index()
        grouped_df = grouped_df[grouped_df['Tau'].apply(lambda x: len(x) > 1)]
        df_list[d] = grouped_df
        Statsdf = pd.concat([df_list[k] for k in range(len(df_list))])
        Statsdf = Statsdf[Statsdf['Tau'].apply(lambda x: isinstance(x, list))]
        statistic, p_value = f_oneway(*Statsdf['Tau'])
    return statistic, p_value, len(Statsdf['Tau'])

def Kruskaltest(df_list, Gipps=False):
    for d in range(len(df_list)):
        if Gipps == True:
            df_list[d]['Tau']=df_list[d]['reaction time']
        df_list[d].dropna(subset=['Tau'], inplace=True)
        grouped_df = df_list[d].groupby('id')['Tau'].agg(list).reset_index()
        grouped_df = grouped_df[grouped_df['Tau'].apply(lambda x: len(x) > 1)]
        df_list[d] = grouped_df
        Statsdf = pd.concat([df_list[k] for k in range(len(df_list))])
        Statsdf = Statsdf[Statsdf['Tau'].apply(lambda x: isinstance(x, list))]
        statistic, p_value = kruskal(*Statsdf['Tau'])
    return statistic, p_value, len(Statsdf['Tau'])

def test_Mixedeffectmodel_linear_global_local_Tau(df_loc_list, df_glob, REML=True, Gipps=False):
    '''
    Performs a mixed-effect model to compare the relationship between global and local Tau values.

    Parameters:
    -----------
    df_loc_list : list
        List of DataFrames containing local Tau values.
    df_glob : DataFrame
        DataFrame containing global Tau values.
    REML : bool, optional
        Flag indicating whether to use REML (Restricted Maximum Likelihood) estimation, by default True.

    Returns:
    --------
    result : MixedLMResults
        Results of the mixed-effect model.
    '''
    for d in range(len(df_loc_list)):
        if Gipps == True:
            df_loc_list[d]['Tau']=df_loc_list[d]['reaction time']
        df_loc_list[d].dropna(subset=['Tau'], inplace=True)
        grouped_df = df_loc_list[d].groupby('id')['Tau'].agg(list).reset_index()
        grouped_df = grouped_df[grouped_df['Tau'].apply(lambda x: len(x) > 1)]
        df_loc_list[d] = grouped_df
        Statsdf = pd.concat([df_loc_list[k] for k in range(len(df_loc_list))])

    # Merge DataFrames on 'id' column
    df_glob['Tau'] = df_glob['reaction time']
    merged_df = pd.merge(Statsdf, df_glob[['id', 'Tau']], on='id')
    merged_df = merged_df.rename(columns={'Tau_y': 'Tau glob', 'Tau_x': 'Tau loc'})
    # Explode the list column into individual values
    exploded_df = merged_df.explode('Tau loc')

    # Rename the column for accessibility in the regression formula
    exploded_df = exploded_df.rename(columns={'Tau loc': 'Tau_loc'})
    exploded_df['Tau_loc'] = pd.to_numeric(exploded_df['Tau_loc'])
    # Specify the mixed-effect regression model
    exploded_df.dropna(inplace=True)
    mixed_model = smf.mixedlm("Tau_loc ~ Q('Tau glob')", exploded_df, groups=exploded_df['id'])

    result = mixed_model.fit(reml=REML)

    return result
