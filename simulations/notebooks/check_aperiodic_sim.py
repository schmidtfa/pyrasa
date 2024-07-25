# %%
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

import scipy.stats as stats

sns.set_style('ticks')
sns.set_context('poster')

# %%
path_list = list(Path('/home/schmidtfa/git/pyrasa/paper/data/knee_sim').glob('*/*.dat'))
# %%

all_dfs = []
for f in path_list:
    f_info = str(f).split('/')[-1].split('_')
    cur_data = pd.concat(joblib.load(f)['ap_fits'])
    cur_data['pre_knee_exp'] = float(f_info[8])
    cur_data['n_seconds'] = int(f_info[-5])
    cur_data['fit_func'] = f_info[4]
    all_dfs.append(cur_data)

# %%
cur_pre = 0.5
n_secs = 300

df_cmb = pd.concat(all_dfs).query(f'pre_knee_exp == {cur_pre}')
exponent = 'Exponent_2'  # or Exponent_2
# %%

# df_cmb['Knee Frequency (Hz)'] = df_cmb['Knee'] ** (1. / (2*cur_pre + df_cmb[exponent]))
df_cmb['relative Error (Knee Frequency (Hz))'] = df_cmb['GT_Knee_Freq'] - df_cmb['Knee Frequency (Hz)']
df_cmb['relative Error (Exponent)'] = cur_pre + df_cmb['GT_Exponent'] - df_cmb[exponent]
df_cmb['Error (Knee Frequency (Hz))'] = np.abs(df_cmb['GT_Knee_Freq'] - df_cmb['Knee Frequency (Hz)'])
df_cmb['Error (Exponent)'] = np.abs(cur_pre + df_cmb['GT_Exponent'] - df_cmb[exponent])
df_cmb = df_cmb[df_cmb['Knee Frequency (Hz)'] < 100]

# %%
sns.catplot(df_cmb, x='n_seconds', y='Error (Knee Frequency (Hz))', hue='param_type', kind='point')
# %%
sns.catplot(df_cmb, x='n_seconds', y='Error (Exponent)', hue='param_type', kind='point')


df_cmb_cut = df_cmb.query(f'n_seconds == {n_secs}').drop(columns='n_seconds')

no_rasa = df_cmb_cut.query('param_type == "norasa"')
irasa = df_cmb_cut.query('param_type == "irasa"')
# %%
cor_cols = [
    'Offset',
    'Exponent_1',
    'Exponent_2',
    'Knee Frequency (Hz)',
    'relative Error (Knee Frequency (Hz))',
    'relative Error (Exponent)',
]

for col in cor_cols:
    print(f'{col}: {stats.pearsonr(irasa[col], no_rasa[col])}')

# %%
df_list = []
for sec in [4, 8, 30, 60, 180, 300]:
    df_cmb_cut = df_cmb.query(f'n_seconds == {sec}').drop(columns='n_seconds')

    no_rasa = df_cmb_cut.query('param_type == "norasa"')
    irasa = df_cmb_cut.query('param_type == "irasa"')

    for col in cor_cols:
        cur_df = pd.DataFrame({'r': stats.pearsonr(irasa[col], no_rasa[col])[0], 'col': col, 'nsecs': sec}, index=[0])
        df_list.append(cur_df)

# %%
df_corrs = pd.concat(df_list)

grid = sns.FacetGrid(df_corrs, col='col', hue='col', palette='deep', col_wrap=3, height=4, aspect=2)

grid.map(plt.plot, 'nsecs', 'r', marker='o')
grid.set(ylim=(0, 1.1))
grid.tight_layout(w_pad=1)

# %%
cols_of_interest = [
    'param_type',
    'Knee',
    'Offset',
    'Exponent_2',
    'Error (Knee Frequency (Hz))',
    'Error (Exponent)',
    'Knee Frequency (Hz)',
]


corr_eff = df_cmb_cut[cols_of_interest].groupby('param_type').corr()

# %%
f, axes = plt.subplots(ncols=2, figsize=(20, 10), sharex=True, sharey=True)

sns.heatmap(corr_eff.query('param_type == "irasa"'), ax=axes[0])
sns.heatmap(corr_eff.query('param_type == "norasa"'), ax=axes[1])

for ax in axes:
    ax.set_ylabel('')

plt.tight_layout()
# %%

f, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

axes[0].set_title('IRASA')
sns.regplot(
    df_cmb_cut.query('param_type == "irasa"'),
    x='Error (Exponent)',
    y='Knee Frequency (Hz)',
    # hue='param_type',
    ax=axes[0],
)

axes[1].set_title('No IRASA')
sns.regplot(
    df_cmb_cut.query('param_type == "norasa"'),
    x='Error (Exponent)',
    y='Knee Frequency (Hz)',
    # hue='param_type',
    ax=axes[1],
)
# %%
f, axes = plt.subplots(ncols=2, figsize=(12, 6), sharex=True, sharey=True)

axes[0].set_title('IRASA')
sns.regplot(
    df_cmb_cut.query('param_type == "irasa"'),
    x='Error (Knee Frequency (Hz))',
    y='Knee Frequency (Hz)',
    # hue='param_type',
    ax=axes[0],
)
axes[1].set_title('No IRASA')
sns.regplot(
    df_cmb_cut.query('param_type == "norasa"'),
    x='Error (Knee Frequency (Hz))',
    y='Knee Frequency (Hz)',
    # hue='param_type',
    ax=axes[1],
)
# %%
