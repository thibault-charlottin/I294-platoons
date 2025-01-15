import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf
import ast

data = pd.read_csv('out/DTW_platoon_summary.csv')
data['ACC'] = data['ACC'].apply(ast.literal_eval)
data['has_ACC'] = [True if True in k else False for k in data['ACC']]
data['has_ACC'] = data['has_ACC'].astype(int)
data['platoon compo'] = data['platoon compo'].apply(ast.literal_eval)
data['survived'] = data['platoon compo'].shift(-1) == data['platoon compo']
data['survived'] = data['survived'].astype(int)
data['log_duration'] = np.log(data['duration'])
data['heavy'] = data['heavy in platoon'].astype(int)
# Ajout de la colonne 'duration'
data['computed_duration'] = 30  # Initialisation à 30
for i in range(1, len(data)):
    if data.loc[i, 'platoon compo'] == data.loc[i - 1, 'platoon compo']:
        data.loc[i, 'computed_duration'] = data.loc[i - 1, 'computed_duration'] + 5

data['log_duration'] = np.log(data['computed_duration'])

formula = (
    "survived ~ "
    "length + has_ACC +  log_duration"
)


logit_model = smf.logit(formula=formula, data=data)
results = logit_model.fit()
print(results.summary())

