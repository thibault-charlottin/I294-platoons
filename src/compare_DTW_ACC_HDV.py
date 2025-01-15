import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

sns.set_theme()
sns.set(font_scale=1.5)
import os
datapath = 'data/by_run/corrected/'
DTWpath = 'out/string_dtw/'
compile = pd.DataFrame()
for f in os.listdir(datapath):
    datadf = pd.read_csv(datapath+f)
    ACC_ids = pd.unique(datadf[datadf['ACC']=='Yes']['ID'])
    dtw_df = pd.read_csv(DTWpath+f)
    dtw_df = pd.merge(dtw_df, datadf, left_on=['id', 'time'], right_on=['ID', 'time'])
    dtw_df['type'] = ['ACC' if k in ACC_ids else 'HDV' for k in dtw_df['id']]
    compile = pd.concat([compile,dtw_df])
print(compile)

compile = compile[compile['DTW string']<1.5]
compile = compile[compile['speed-kf']<25]
sns.boxplot(data=compile, x='type', y='DTW string', palette={'ACC': 'orange', 'HDV': 'blue'})
plt.tight_layout()
plt.title('DTW values, speed<25m/s')
#plt.savefig('DTW_values distrib.pdf')
plt.show()

sns.boxplot(data=compile, x='type', y='speed-kf', palette={'ACC': 'orange', 'HDV': 'blue'})
plt.title('speed (truncated at speed 25m/s)')
plt.tight_layout()
#plt.savefig('DTW_values distrib.pdf')
plt.show()

ACC_values = compile[compile['type'] == 'ACC']['DTW string']
HDV_values = compile[compile['type'] == 'HDV']['DTW string']
stat, p_value = mannwhitneyu(ACC_values, HDV_values, alternative='less')
print(f"Test Mann-Whitney U unilatéral: stat = {stat}, p = {p_value}")

ACC_values = compile[compile['type'] == 'ACC']['speed-kf']
HDV_values = compile[compile['type'] == 'HDV']['speed-kf']
stat, p_value = mannwhitneyu(ACC_values, HDV_values, alternative='less')
print(f"Test Mann-Whitney U unilatéral: stat = {stat}, p = {p_value}")