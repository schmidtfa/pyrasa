#%%
import joblib
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from pathlib import Path
import numpy as np

import scipy.stats as stats

sns.set_style('ticks')
sns.set_context('poster')

#%%
path_list = list(Path('/home/schmidtfa/git/pyrasa/paper/data/knee_sim').glob('*/*.dat'))
# %%

all_dfs = []
for f in path_list:
    f_info = str(f).split('/')[-1].split('_')
    cur_data = pd.concat(joblib.load(f)['ap_fits'])
    cur_data['n_seconds'] = int(f_info[-5])
    cur_data['fit_func'] = f_info[4]
    all_dfs.append(cur_data)

# %%
cur_fit_func = "double"
df_cmb = pd.concat(all_dfs).query(f'fit_func == "{cur_fit_func}"')
exponent = 'Exponent_2' #or Exponent_2
# %%
if cur_fit_func == 'double':
    df_cmb['Knee Frequency (Hz)'] = df_cmb['Knee'] ** (1. / df_cmb[exponent])
df_cmb['delta_knee'] = np.abs(df_cmb['GT_Knee_Freq'] - df_cmb['Knee Frequency (Hz)'])
df_cmb['delta_exponent'] = np.abs(df_cmb['GT_Exponent'] - df_cmb[exponent]) #if piecewise
# %%
sns.catplot(df_cmb, x='n_seconds', y='delta_knee', 
            hue='param_type', kind='point') 
# %%
sns.catplot(df_cmb, x='n_seconds', 
            y='delta_exponent', 
            hue='param_type', kind='point') 

# %%
cols_of_interest = ['param_type', 'Knee', 'Offset', 'Exponent_2', 
                    'n_seconds', 'delta_knee', 'delta_exponent', 'Knee Frequency (Hz)']


df_cmb[cols_of_interest].groupby('param_type').corr()
# %%

f, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

sns.regplot(df_cmb.query('param_type == "irasa"'), x='delta_exponent',
                 y='Knee Frequency (Hz)', 
                 #hue='param_type',
                 ax=axes[0])

sns.regplot(df_cmb.query('param_type == "norasa"'), x='delta_exponent',
                 y='Knee Frequency (Hz)', 
                 #hue='param_type',
                 ax=axes[1])
# %%
f, axes = plt.subplots(ncols=2, figsize=(12, 6), sharey=True)

sns.regplot(df_cmb.query('param_type == "irasa"'), x='delta_knee',
                 y='Knee Frequency (Hz)', 
                 #hue='param_type',
                 ax=axes[0])

sns.regplot(df_cmb.query('param_type == "norasa"'), x='delta_knee',
                 y='Knee Frequency (Hz)', 
                 #hue='param_type',
                 ax=axes[1])
# %%
