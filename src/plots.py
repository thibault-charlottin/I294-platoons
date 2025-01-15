import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme("notebook")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import os
import re

def extract_file_number(filename):
    '''
    Extracts the numerical part from a filename formatted as 'runX.csv', where X is the file number.

    Parameters:
    -----------
    filename : str
        Name of the file.

    Returns:
    --------
    int or None
        Extracted file number if found, otherwise None.
    '''
    match = re.search(r'run(\d+)\.csv', filename)
    if match:
        return int(match.group(1))
    else:
        return None
    
def plot_Xt(df, file, subpart=False, lane_subpart=0, begin_subpart=0, end_subpart=0,savelane=False, saveall=False, plot_ACC= False):
    '''
    Plots Xt (position vs time) for each lane and optionally saves the plots.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data.
    file : str
        Name of the file.
    subpart : bool, optional
        Flag to indicate if plotting a subpart of the data, by default False.
    lane_subpart : int, optional
        Lane number for the subpart, by default 0.
    begin_subpart : int, optional
        Start time for the subpart, by default 0.
    end_subpart : int, optional
        End time for the subpart, by default 0.
    savelane : bool, optional
        Flag to save individual lane plots, by default False.
    saveall : bool, optional
        Flag to save the entire plot, by default False.
    plot_ACC : bool, optional
        Flag to plot ACC trajectory, by default False.
    '''
    if subpart:
        df = df[(df['time'] > begin_subpart) & (df['time'] < end_subpart)]
        df = df[df['lane-kf'] == lane_subpart]
    file = extract_file_number(file)
    # Définition des couleurs personnalisées
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Rouge, Jaune, Vert
    n_bins = 100  # Nombre de bins pour l'interpolation
    cmap = LinearSegmentedColormap.from_list('custom_colormap', colors, N=n_bins)
    norm = matplotlib.colors.Normalize(0, 40)
    
    # Obtenir le nombre de valeurs uniques de 'lane-kf'
    unique_lanes = pd.unique(df['lane-kf'])
    num_lanes = len(unique_lanes)
    print(num_lanes)
    # Créer une figure et des sous-tracés dynamiquement
    if num_lanes > 1:
        fig, axes = plt.subplots(1, num_lanes, figsize=(5*num_lanes, 5))

        for i, lane_id in enumerate(unique_lanes):
            lane_df = df[df['lane-kf'] == lane_id]
            for track_id, track_data in lane_df.groupby('ID'):
                axes[i].scatter(track_data['time'], track_data['r'], c=track_data['speed-kf'], norm=norm, s=0.2, cmap=cmap)
            axes[i].set_title(f'Lane {lane_id}')
            axes[i].set_xlabel('Time [s]')
            axes[i].set_ylabel('X [m]')
            axes[i].grid(True)
            axes[i].set_aspect('auto')
            axes[i].margins(0.05)
            if plot_ACC:
                ACC = lane_df[lane_df['ACC']=='Yes']
                axes[i].scatter(ACC['time'],ACC['r'],s=0.2,label = 'ACC trajectory')
                axes[i].legend()
            if savelane:
                plt.savefig(f'out/images/XT/by_lane/Xt of run {file} for lane {lane_id}.pdf')
    else : 
        fig, axes = plt.subplots(1, num_lanes, figsize=(5*num_lanes, 5))

        for i, lane_id in enumerate(unique_lanes):
            lane_df = df[df['lane-kf'] == lane_id]
            for track_id, track_data in lane_df.groupby('ID'):
                axes.scatter(track_data['time'], track_data['r'], c=track_data['speed-kf'], norm=norm, s=0.2, cmap=cmap)
            axes.set_title(f'Lane {lane_id}')
            axes.set_xlabel('Time [s]')
            axes.set_ylabel('X [m]')
            axes.grid(True)
            axes.set_aspect('auto')
            axes.margins(0.05)
            if plot_ACC:
                ACC = lane_df[lane_df['ACC']=='Yes']
                axes.scatter(ACC['time'],ACC['r'],s=0.2,label = 'ACC trajectory')
                axes.legend()
            if savelane:
                plt.savefig(f'out/images/XT/by_lane/Xt of run {file} for lane {lane_id}.pdf')
        # Ajouter une barre de couleur commune
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([1.02, 0.15, 0.01, 0.7])
        fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbar_ax, label='speed [m/s]')
        plt.suptitle(f'Xt for run {file}', fontsize=16)
        plt.tight_layout()
    if saveall:
        if plot_ACC:
            plt.savefig(f'out/images/XT/global/Xt of run {file} with ACC trajectory.pdf')
        else:
            plt.savefig(f'out/images/XT/global/Xt of run {file}.pdf')
    plt.show()
    return



def plot_cumulative_distribution(path, key, ACC = False):
    '''
    Plots the cumulative distribution of a specified key from files in a directory.

    Parameters:
    -----------
    path : str
        Directory path containing files.
    key : str
        Column name for plotting.
    ACC : bool, optional
        Flag to include only ACC data, by default False.
    '''
    for run in os.listdir(path):
        df = pd.read_csv(path+run)
        df = df[df['THW']<10] 
        if ACC ==True:
            df = df[df['ACC']=='Yes']
        sns.ecdfplot(data=df, x=key, label = run)
    plt.legend(loc = 'lower right')
    if '-kf' in key : 
        key = key[0:-3]
    if ACC ==True:
        plt.title(f'Cumulative plot of {key}\n ACC only')
        plt.savefig(f'Cumulative plot of  {key} ACC only.pdf')
        return
    plt.title(f'out/images/Cumulative plot of {key}\n all vehicles')
    plt.savefig(f'out/images/Cumulative plot of  {key} all vehicles.pdf')
    return

def plot_Tau_distribution(path, save = False, type='Newell', Gipps=False):
    '''
    Plots the distribution of Tau values from files in a directory.

    Parameters:
    -----------
    path : str
        Directory path containing files.
    save : bool, optional
        Flag to save the plot, by default False.
    type : str, optional
        Type of plot (Local or Global), by default 'Local'.

    Returns:
    --------
    DataFrame
        Concatenated DataFrame containing all Tau values.
    '''
    files = os.listdir(path)
    df_out = pd.DataFrame({'id':[],'leader':[],'Tau':[],'D':[]})
    for f in files : 
        df = pd.read_csv(path + '/' +f)
        if Gipps==True:
            df ['Tau'] = df['reaction time']
        df_out = pd.concat([df_out,df])
    df_out.reset_index(inplace=True)
    sns.histplot(data = df_out, x = 'Tau', edgecolor='none', label = 'mean \u03C4 = '+str(np.round(np.mean(df_out['Tau']),2)))
    plt.legend()
    if save == True:
        plt.savefig(f'out/images/{type} Tau distribution.pdf')
    return df_out

def plot_Delta_distribution(path, save = False, type='Newell', Gipps=False):
    '''
    Plots the distribution of Delta (D) values from files in a directory.

    Parameters:
    -----------
    path : str
        Directory path containing files.
    save : bool, optional
        Flag to save the plot, by default False.
    type : str, optional
        Type of plot (Local or Global), by default 'Local'.

    Returns:
    --------
    DataFrame
        Concatenated DataFrame containing all D values.
    '''
    files = os.listdir(path)
    df_out = pd.DataFrame({'id':[],'leader':[],'Tau':[],'D':[]})
    for f in files : 
        df = pd.read_csv(path + '/' +f)
        if Gipps==True:
            df['D']=df['dist. stand hill']
        df_out = pd.concat([df_out,df])
    df_out.reset_index(inplace=True)
    sns.histplot(data = df_out, x = 'D', label = 'mean \u03B4 = '+str(np.round(np.mean(df_out['D']),2)))
    plt.legend()
    if save == True:
        plt.savefig(f'out/images/{type} Delta distribution.pdf')
    return df_out


def plot_mixed_effect_models(resultML, resultREML, save = False):
    '''
    Plots the mixed effect models.

    Parameters:
    -----------
    resultML : MixedLMResults
        Results of the ML model.
    resultREML : MixedLMResults
        Results of the REML model.
    save : bool, optional
        Flag to save the plots, by default False.
    '''
    # Plot for ML model
    plt.figure(figsize=(10, 6))
    plt.scatter(resultML.model.exog[:, 1], resultML.model.endog,marker = '+', color = 'black', label='Data')
    abline_values = [resultML.params['Intercept'] + resultML.params['Q(\'Tau glob\')'] * x for x in resultML.model.exog[:, 1]]
    plt.plot(resultML.model.exog[:, 1], abline_values, color='red', label='ML Model')
    
    # Calculate marginal means
    marginal_means_inf = resultML.fe_params['Intercept'] + (resultML.fe_params['Q(\'Tau glob\')'] - np.unique(resultML.model.exog[:, 1])[0]) * np.unique(resultML.model.exog[:, 1])
    plt.plot(np.unique(resultML.model.exog[:, 1]), marginal_means_inf, color='red', linestyle='dashed', label='Marginal Means')
    marginal_means_sup = resultML.fe_params['Intercept'] + (resultML.fe_params['Q(\'Tau glob\')'] + np.unique(resultML.model.exog[:, 1])[1]) * np.unique(resultML.model.exog[:, 1])
    plt.plot(np.unique(resultML.model.exog[:, 1]), marginal_means_sup, color='red', linestyle='dashed')
    plt.fill_between(np.unique(resultML.model.exog[:, 1]), marginal_means_inf, marginal_means_sup, color='red', alpha=0.2)
    plt.xlabel('Tau computed on entire trajectories')
    plt.ylabel('Tau computed by segment')
    plt.title('Mixed Effect Model (ML)')
    plt.legend()
    if save==True:
        plt.savefig(f'out/images/ML model.pdf')
    plt.show()

    # Plot for REML model
    plt.figure(figsize=(10, 6))
    plt.scatter(resultREML.model.exog[:, 1], resultREML.model.endog,marker = '+', color = 'black', label='Data')
    abline_values = [resultREML.params['Intercept'] + resultREML.params['Q(\'Tau glob\')'] * x for x in resultREML.model.exog[:, 1]]
    plt.plot(resultREML.model.exog[:, 1], abline_values, color='green', label='REML Model')
    
    # Calculate marginal means
    marginal_means_inf = resultREML.fe_params['Intercept'] + (resultREML.fe_params['Q(\'Tau glob\')'] - np.unique(resultREML.model.exog[:, 1])[0]) * np.unique(resultREML.model.exog[:, 1])
    plt.plot(np.unique(resultML.model.exog[:, 1]), marginal_means_inf, color='green', linestyle='dashed', label='Marginal Means')
    marginal_means_sup = resultREML.fe_params['Intercept'] + (resultREML.fe_params['Q(\'Tau glob\')'] + np.unique(resultREML.model.exog[:, 1])[1]) * np.unique(resultREML.model.exog[:, 1])
    plt.plot(np.unique(resultREML.model.exog[:, 1]), marginal_means_sup, color='green', linestyle='dashed')
    plt.fill_between(np.unique(resultREML.model.exog[:, 1]), marginal_means_inf, marginal_means_sup, color='green', alpha=0.2)
    plt.xlabel('Tau glob')
    plt.ylabel('Tau loc')
    plt.title('Mixed Effect Model (REML)')
    plt.legend()
    if save==True:
        plt.savefig(f'out/images/REML model.pdf')
    plt.show()

def plot_quality_distribution(path_glob,path_loc,save = False):
    '''
    Plots the comparison of quality distribution between global and local Tau calculations.

    Parameters:
    -----------
    path_glob : str
        Directory path containing files for global Tau calculation.
    path_loc : str
        Directory path containing files for local Tau calculation.
    save : bool, optional
        Flag to save the plots, by default False.
    '''
    files = os.listdir(path_glob)
    df_glob = pd.DataFrame({'id':[],'leader':[],'Ie':[],'Ip':[]})
    for f in files : 
        df = pd.read_csv(path_glob + '/' +f)
        df_glob = pd.concat([df_glob,df])
    df_glob.reset_index(inplace=True)
    df_glob['Ip'] = np.abs(df_glob['Ip'])
    df_glob['Ie'] = np.abs(df_glob['Ie'])
    files = os.listdir(path_loc)
    df_loc = pd.DataFrame({'id':[],'leader':[],'Ie':[],'Ip':[]})
    for f in files : 
        df = pd.read_csv(path_loc + '/' +f)
        df_loc = pd.concat([df_loc,df])
    df_loc.reset_index(inplace=True)
    df_loc['Ip'] = np.abs(df_loc['Ip'])
    df_loc['Ie'] = np.abs(df_loc['Ie'])
    sns.histplot(data = df_glob, x = 'Ie', kde = True, label = 'global Tau calulation\n mean Ie = '+str(np.mean(df_glob['Ie'])))
    sns.histplot(data = df_loc, x = 'Ie', kde = True, label = 'local Tau calulation\n mean Ie = '+str(np.mean(df_loc['Ie'])))
    plt.legend()
    if save==True:
        plt.savefig(f'out/images/Ie comparison.pdf')
    plt.show()
    sns.histplot(data = df_glob, x = 'Ip', kde = True, label = 'global Tau calulation\n mean Ip = '+str(np.mean(df_glob['Ip'])))
    sns.histplot(data = df_loc, x = 'Ip', kde = True, label = 'local Tau calulation\n mean Ie = '+str(np.mean(df_loc['Ip'])))
    plt.legend()
    if save==True:
        plt.savefig(f'out/images/Ip comparison.pdf')
    plt.show()
    return 