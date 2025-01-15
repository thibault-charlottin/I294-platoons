import pandas as pd
import statsmodels.api as sm
import numpy as np
import ast
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import seaborn as sns
sns.set_theme()
sns.set(font_scale=1.5)
def count_platoons_per_duration(data_platoons):
    number = []
    duration = []
    for t in pd.unique(data_platoons['duration']):
        test = data_platoons[data_platoons['duration'] >= t]
        number.append(len(test['duration']))
        duration.append(t)
    return pd.DataFrame({'number of platoons': number, 'duration': duration}).sort_values(by='duration', ascending=True)


data_platoons = pd.read_csv('out/DTW_platoon_summary.csv')
data_platoons['duration'] = 30  # Initialisation à 30
data_platoons['platoon compo'] = data_platoons['platoon compo'].apply(ast.literal_eval)
for i in range(1, len(data_platoons)):
    if data_platoons.loc[i, 'platoon compo'] == data_platoons.loc[i - 1, 'platoon compo']:
        data_platoons.loc[i, 'duration'] = data_platoons.loc[i - 1, 'duration'] + 5
data_platoons['ACC'] = data_platoons['ACC'].apply(ast.literal_eval)
data_platoons['has_ACC'] = [True if True in k else False for k in data_platoons['ACC']]
print(data_platoons['has_ACC'])




data_no_ACC = data_platoons[data_platoons['has_ACC']==False]
data_no_ACC = count_platoons_per_duration(data_no_ACC)
data_no_ACC['log_remaining_platoons'] = np.log(data_no_ACC['number of platoons'])


data_ACC = count_platoons_per_duration(data_platoons)
data_ACC['log_remaining_platoons'] = np.log(data_ACC['number of platoons'])

x = sm.add_constant(data_no_ACC['duration'])  
Y = data_no_ACC['log_remaining_platoons']
model = sm.OLS(Y, x).fit()
residuals = model.resid
print('No ACC')
stat, p = shapiro(residuals)
print('Statistique de Shapiro-Wilk:', stat)
print('P-valeur de Shapiro-Wilk:', p)
print(model.summary())

X = sm.add_constant(data_ACC['duration'])  
y = data_ACC['log_remaining_platoons']
model_ACC = sm.OLS(y, X).fit()
residuals = model_ACC.resid
print('ACC')
stat, p = shapiro(residuals)
print('Statistique de Shapiro-Wilk:', stat)
print('P-valeur de Shapiro-Wilk:', p)
print(model_ACC.summary())
intercept = 6.3966
slope_ACC = -0.0462
conf_int_ACC = np.exp(np.array([-0.051,-0.042]))
x_line = np.linspace(0, max(data_ACC['duration']), 100)
y_line = np.exp(slope_ACC * x_line +intercept)
lower_bound = conf_int_ACC[0] * x_line
upper_bound = conf_int_ACC[1] * x_line
fig = plt.figure(figsize = (8,5))
plt.scatter(data_ACC['duration'],data_ACC['number of platoons'],color = 'blue', label ='all platoons')
plt.scatter(data_no_ACC['duration'],data_no_ACC['number of platoons'],color = 'orange', label='HDV platoons')



plt.xlabel('duration [s]')
plt.ylabel('number of platoons')
plt.legend()
plt.tight_layout()
plt.savefig('remaining_TGSIM_L1_platoons.pdf')
plt.show()