# %%
import joblib
from pathlib import Path
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')


base_p = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1])

all_files = list(Path(base_p + '/data/sim_all_f/knee').glob('knee__*.dat'))
# %%

ap_list, p_list = [], []

for f in all_files:
    cur_f = joblib.load(f)

    cur_p = pd.concat([cur_f['specparam']['knee']['periodic'], cur_f['pyrasa']['knee']['periodic']])  # .dropna(axis=1)

    cur_ap = pd.concat(
        [cur_f['specparam']['knee']['aperiodic'], cur_f['pyrasa']['knee']['aperiodic']]
    )  # .dropna(axis=1)

    ap_list.append(cur_ap)
    p_list.append(cur_p)

# %%
df_ap_sim = pd.concat(ap_list)

# %%


# %%
df_ap_sim['sim_exp'] = df_ap_sim['sim_exp'].abs()
df_ap_sim = df_ap_sim.query('sim_exp > 1')
df_ap_sim['Exponent Δ'] = np.abs(df_ap_sim['Exponent_2'] - df_ap_sim['sim_exp'])

df_ap_sim['sim_kf'] = df_ap_sim['sim_knee'] ** (1.0 / df_ap_sim['sim_exp'])
# %%
knee_logical = np.logical_and(df_ap_sim['sim_kf'] > 1, df_ap_sim['sim_kf'] < 80)
df_ap_sim = df_ap_sim[knee_logical]

df_ap_sim['Knee Frequency Δ'] = np.abs(df_ap_sim['Knee Frequency (Hz)'] - df_ap_sim['sim_kf'])
# df_ap_sim = df_ap_sim[df_ap_sim['Knee Δ'] < 100]

# %%
palette = 'deep'

f, ax = plt.subplots(figsize=(4, 4))
sns.barplot(
    df_ap_sim,  # dropna().reset_index(),
    x='method',
    y='Knee Frequency Δ',
    palette=palette,
    hue='method',
    # split=True,
    legend=False,
    ax=ax,
)

sns.despine()
ax.set_xlabel('')
f.savefig('../results/knee_freq_delta_sim.svg')

# %%
f, ax = plt.subplots(figsize=(4, 4))
sns.barplot(
    df_ap_sim,
    x='method',
    y='Exponent Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=ax,
)

sns.despine()
f.savefig('../results/exp_delta_sim.svg')

# %%
f, ax = plt.subplots(figsize=(8, 10), nrows=3)
sns.lineplot(df_ap_sim, x='sim_freq', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[0])
ax[0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_ap_sim, x='sim_exp', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[1])

ax[1].set_xlabel('Simulated Exponent')

sns.lineplot(df_ap_sim, x='sim_kf', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[2])

ax[2].set_xlabel('Simulated Knee Frequency (Hz)')


sns.despine()
f.tight_layout()

# %%
df_p_sim = pd.concat(p_list)
df_p_sim['sim_exp'] = np.abs(df_p_sim['sim_exp'])
df_p_sim = df_p_sim.query('sim_exp > 1')
# %%
df_p_sim['sim_kf'] = df_p_sim['sim_knee'] ** (1.0 / df_p_sim['sim_exp'])
knee_logical = np.logical_and(df_p_sim['sim_kf'] > 2, df_p_sim['sim_kf'] < 50)
df_p_sim = df_p_sim[knee_logical]

# %%
df_nona = df_p_sim.copy().query('cf < 40')
df_nona['Center \n Frequency Δ'] = np.abs(df_nona['cf'] - df_nona['sim_freq'])
# %%
f, ax = plt.subplots(figsize=(4, 6))
sns.violinplot(
    df_nona,
    x='method',
    y='Center \n Frequency Δ',
    hue='method',
    # split=True,
    legend=False,
    ax=ax,
)
# ax.set_ylabel('Exponent Δ')
ax.set_xlabel('')
sns.despine()
# %%
f, ax = plt.subplots(figsize=(8, 10), nrows=3)
sns.lineplot(df_nona, x='sim_freq', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0])
ax[0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_nona, x='sim_exp', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1])

ax[1].set_xlabel('Simulated Exponent')

sns.lineplot(df_nona, x='sim_knee', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[2])

ax[2].set_xlabel('Simulated Knee')

sns.despine()
f.tight_layout()

# %%

df_hp_sp = df_p_sim.query('method == "specparam"').query('pw > 1').copy()
df_ir = df_p_sim.query('method == "pyrasa"')

df_p_cmb = pd.concat([df_hp_sp, df_ir])

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
    y='Center \n Frequency Δ',
    hue='method',
    palette=palette,
    # split=True,
    legend=False,
    ax=axes[1],
)

sns.violinplot(
    df_p_cmb,
    x='method',
    y='Center \n Frequency Δ',
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
