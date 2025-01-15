import pandas as pd
import ast
import os
import numpy as np
import scipy.stats as st
df = pd.read_csv('out/DTW_platoon_summary.csv')

strings_dtw_path = 'out/DTW/'


df['platoon DTW'] = df['platoon DTW'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
df['platoon compo'] = df['platoon compo'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])
list_dtw_string = []
for run in pd.unique(df['run']):
    DTW_df = pd.read_csv(strings_dtw_path+'DTW_'+run)
    loc_df = df[df['run']==run]
    loc_df.reset_index(inplace=True,drop=True)
    for k in range(len(loc_df['platoon compo'])):
        dtw_test = DTW_df[(DTW_df['id'].isin(loc_df['platoon compo'][k]))&(DTW_df['time']==loc_df['time'][k])]
        list_dtw_string.append(pd.unique(dtw_test['DTW']))
    
df['CF DTW'] = list_dtw_string

diff = []
for k in range(len(df['run'])):
    platoon = np.array(df['platoon DTW'][k])
    CF = np.array(df['CF DTW'][k])
    if len(platoon)==len(CF):
        diff_array = np.abs(CF-platoon)
    else :
        diff_array = [np.nan]
    diff_array[0]=0
    diff.append(diff_array)
df['difference']=diff
df.to_csv('out/platoon_DTW_CF_DTW.csv')

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


dtw_data =  []
for idx, row in df.iterrows():
    for rank, value in enumerate(row['CF DTW']):
        # Set CF DTW to match platoon DTW if rank is 0
        if rank == 0:
            value = row['platoon DTW'][0]
        dtw_data.append({"Rank": rank, "Value": value, "Category": "CF DTW"})
    
    for rank, value in enumerate(row['platoon DTW']):
        dtw_data.append({"Rank": rank, "Value": value, "Category": "Platoon DTW"})

# Create a DataFrame from the adjusted data
dtw_df = pd.DataFrame(dtw_data)
dtw_df['Rank'] = dtw_df['Rank']+1
dtw_df = dtw_df[dtw_df['Rank']<=10]
# Plotting boxplots for both 'CF DTW' and 'platoon DTW' on the same figure
plt.figure(figsize=(12, 6))
sns.boxplot(x="Rank", y="Value", hue="Category",palette=['blue','orange'], data=dtw_df, showfliers = False)
plt.xlabel("Position in platoon", size = 15)
plt.ylabel("DTW Value", size = 15)
plt.legend(fontsize = 13)
plt.xticks([0,1,2,3,4,5,6,7,8,9,10],size = 13)
plt.yticks([0,0.3,0.6,0.9,1.2,1.5],size = 13)

plt.show()
# Initialisation des listes pour stocker les valeurs de la statistique t, des p-values et des couleurs
t_statistics = []
max_ranks_tested = []
colors = []

# Boucle pour augmenter max_rank et effectuer le test de différence entre CF DTW et Platoon DTW
max_rank = 2
while max_rank <= dtw_df['Rank'].max():
    # Filtrer les données jusqu'à max_rank
    filtered_df = dtw_df[dtw_df['Rank'] == max_rank]
    
    # Séparer les valeurs CF DTW et Platoon DTW
    cf_dtw_values = filtered_df[filtered_df['Category'] == "CF DTW"]['Value']
    platoon_dtw_values = filtered_df[filtered_df['Category'] == "Platoon DTW"]['Value']
    
    # Effectuer un test t unilatéral
    t_stat, p_value = st.ttest_ind(platoon_dtw_values, cf_dtw_values, alternative='greater', equal_var=False)
    
    # Stocker la valeur de la statistique t et la couleur en fonction de la p-valeur
    t_statistics.append(t_stat)
    max_ranks_tested.append(max_rank)
    if p_value < 0.05:
        colors.append('green')  # Test significatif
    else:
        colors.append('red')  # Test non significatif
    
    # Incrémenter max_rank
    max_rank += 1

# Tracer les valeurs de la statistique t en fonction de max_rank testé avec couleur basée sur p-valeur
plt.figure(figsize=(10, 6))
for i in range(len(max_ranks_tested)):
    plt.scatter(max_ranks_tested[i], t_statistics[i], color=colors[i], s=100)  # Points colorés par p-valeur

plt.xlabel("Position", size=15)
plt.ylabel("t-Statistic Value", size=15)
plt.xticks(size=13)
plt.yticks(size=13)
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.7)  # Ligne pour y=0
plt.show()
import statsmodels.api as sm
import scipy.stats as st

difference_data = []
for idx, row in df.iterrows():
    for rank, value in enumerate(row['difference']):
        difference_data.append({"Rank": rank + 1, "Difference": value})

difference_df = pd.DataFrame(difference_data)

# Initialisation des listes pour stocker les valeurs de beta et de couleurs
beta_values = []
max_ranks_tested = []
colors = []  # Stocke les couleurs en fonction de la normalité des résidus

# Boucle pour augmenter max_rank et effectuer la régression
max_rank = 2
while max_rank <= max(difference_df['Rank']):
    # Filtrer les données jusqu'à la valeur de max_rank
    filtered_df = difference_df[difference_df['Rank'] <= max_rank]
    
    # Préparation des données pour la régression linéaire
    X = sm.add_constant(filtered_df['Rank'])
    y = filtered_df['Difference']
    
    # Effectuer la régression linéaire
    model = sm.OLS(y, X).fit()
    beta_rank = model.params['Rank']
    
    # Calcul des résidus
    residuals = model.resid
    
    # Test de normalité des résidus (Shapiro-Wilk)
    shapiro_test = st.shapiro(residuals)
    if shapiro_test.pvalue > 0.05:
        colors.append('green')  # Normalité acceptée
    else:
        colors.append('red')  # Normalité rejetée

    # Stocker les valeurs de beta et de max_rank testé
    beta_values.append(beta_rank)
    max_ranks_tested.append(max_rank)
    
    # Incrémenter max_rank
    max_rank += 1

# Tracer les valeurs de beta en fonction du max_rank testé avec couleurs basées sur la normalité des résidus
plt.figure(figsize=(10, 6))
for i in range(len(max_ranks_tested)):
    plt.plot(max_ranks_tested[i], beta_values[i], marker='o', color=colors[i])

plt.xlabel("Max Rank Tested", size=15)
plt.ylabel("Beta Value of Rank", size=15)
plt.title("Beta Value of Rank in Linear Regression of Difference vs Rank\n(Green=Normal Residuals, Red=Non-Normal Residuals)")
plt.grid(True)
plt.show()