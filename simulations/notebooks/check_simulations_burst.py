# %%
import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')


base_p = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1])

all_files = list(Path(base_p + '/data/sim_all_f/burst').glob('*.dat'))
# %%

ap_list, p_list = [], []

for f in all_files:
    cur_f = joblib.load(f)

    cur_p = pd.concat(
        [cur_f['specparam']['fixed']['periodic'], cur_f['pyrasa']['fixed']['periodic']]
    )  # .dropna(axis=1)

    cur_ap = pd.concat(
        [cur_f['specparam']['fixed']['aperiodic'], cur_f['pyrasa']['fixed']['aperiodic']]
    )  # .dropna(axis=1)

    ap_list.append(cur_ap)
    p_list.append(cur_p)

# %%
df_ap_sim = pd.concat(ap_list).reset_index()


# %%
palette = 'deep'
df_ap_sim['sim_exp'] = df_ap_sim['sim_exp'].abs()
df_ap_sim['Exponent Δ'] = np.abs(df_ap_sim['Exponent'] - df_ap_sim['sim_exp'])
# %%
f, ax = plt.subplots(figsize=(8, 14), nrows=3)
sns.lineplot(df_ap_sim, x='sim_freq', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[0])
ax[0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_ap_sim, x='sim_exp', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[1])

ax[1].set_xlabel('Simulated Exponent')

sns.lineplot(df_ap_sim, x='sim_b_proba', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[2])

ax[2].set_xlabel('Simulated Burst Probability')

sns.despine()
f.tight_layout()

# %%
df_p_sim = pd.concat(p_list)
df_p_sim['sim_exp'] = df_p_sim['sim_exp'].abs()
# %%
df_nona = df_p_sim.copy().query('cf < 40').reset_index()
df_nona['Center Frequency Δ'] = np.abs(df_nona['cf'] - df_nona['sim_freq'])

# %%
f, ax = plt.subplots(figsize=(4, 6))
sns.violinplot(
    df_nona,
    x='method',
    y='Center Frequency Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=ax,
)

ax.set_xlabel('')
sns.despine()

# %%
palette = 'deep'

f, ax = plt.subplots(figsize=(6, 14), nrows=3)
sns.lineplot(df_nona, x='sim_freq', y='Center Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0])
ax[0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_nona, x='sim_exp', y='Center Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1])

ax[1].set_xlabel('Simulated Exponent')

sns.lineplot(df_nona, x='sim_b_proba', y='Center Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[2])

ax[2].set_xlabel('Simulated Burst Probability')

sns.despine()
f.tight_layout()
# %%

df_hp_sp = df_nona.query('method == "specparam"').query('pw > 1').copy()
df_ir = df_nona.query('method == "pyrasa"')

df_p_cmb = pd.concat([df_hp_sp, df_ir]).reset_index()

df_p_cmb['Center \n Frequency Δ'] = np.abs(df_p_cmb['cf'] - df_p_cmb['sim_freq'])

f, ax = plt.subplots(figsize=(4, 6))
sns.violinplot(
    df_p_cmb,
    x='method',
    y='Center \n Frequency Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=ax,
)

ax.set_xlabel('')
sns.despine()

# %%
f, ax = plt.subplots(figsize=(8, 10), nrows=3)
sns.lineplot(df_p_cmb, x='sim_freq', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0])
ax[0].set_xlabel('Simulated Oscillation')

sns.lineplot(df_p_cmb, x='sim_exp', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1])

ax[1].set_xlabel('Simulated Exponent')

sns.lineplot(
    df_p_cmb, x='sim_b_proba', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[2]
)

ax[2].set_xlabel('Simulated Burst Probability')

sns.despine()
f.tight_layout()


# %%
f, axes = plt.subplots(ncols=3, figsize=(14, 6))

sns.violinplot(
    df_ap_sim,
    x='method',
    y='Exponent Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=axes[0],
)

sns.violinplot(
    df_nona,
    x='method',
    y='Center Frequency Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=axes[1],
)

sns.violinplot(
    df_p_cmb,
    x='method',
    y='Center Frequency Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=axes[2],
)

titles = ['', 'all peaks', 'specparam \n peaks > 1 (arbitrary)']
for ix, ax in enumerate(axes):
    ax.set_xlabel('')
    ax.set_title(titles[ix])

f.tight_layout()

sns.despine()
# %%
