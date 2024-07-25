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

all_files = list(Path(base_p + '/data/sim_all_f/broad_fixed').glob('*.dat'))
# %%
reload = True
if reload:
    df_ap_sim = pd.read_csv('../results/sim_broad_freq_fixed_ap.csv')
    df_p_sim = pd.read_csv('../results/sim_broad_freq_fixed_p.csv')

else:
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

    df_ap_sim = pd.concat(ap_list).reset_index()
    df_ap_sim['sim_exp'] = df_ap_sim['sim_exp'].abs()
    df_ap_sim.to_csv('../results/sim_broad_freq_fixed_ap.csv')

    df_p_sim = pd.concat(p_list)
    df_p_sim['sim_exp'] = df_p_sim['sim_exp'].abs()

    df_p_sim.to_csv('../results/sim_broad_freq_fixed_p.csv')

# %%

df_ap_sim['sim_exp'] = df_ap_sim['sim_exp'].abs()
df_ap_sim['Exponent Δ'] = np.abs(df_ap_sim['Exponent'] - df_ap_sim['sim_exp'])
# %%
palette = 'deep'

f, ax = plt.subplots(figsize=(14, 14), ncols=2, nrows=2)
sns.lineplot(df_ap_sim, x='sim_freq', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[0, 0])
ax[0, 0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_ap_sim, x='sim_exp', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[0, 1])

ax[0, 1].set_xlabel('Simulated Exponent')

sns.lineplot(df_ap_sim, x='sim_bw', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[1, 0])

ax[1, 0].set_xlabel('Simulated Oscillation Width')

sns.lineplot(df_ap_sim, x='sim_height', y='Exponent Δ', hue='method', palette=palette, legend=False, ax=ax[1, 1])

ax[1, 1].set_xlabel('Simulated Oscillation Height')

sns.despine()
f.tight_layout()

# %%
df_pivot = df_ap_sim.reset_index().pivot_table(index=['method', 'sim_bw'], columns='sim_height', values='Exponent Δ')

# %%


def plot_mesh_bw_h(data, key, ax):
    cur_d = data.copy()
    data2plot = cur_d.query(f'method == "{key}"').to_numpy()
    x_labels = cur_d.query(f'method == "{key}"').columns
    y_labels = cur_d.query(f'method == "{key}"').droplevel('method').index
    title = key
    extent = [
        y_labels.min(),
        y_labels.max(),
        x_labels.min(),
        x_labels.max(),
    ]

    mesh = ax.imshow(
        data2plot, aspect=1.4, vmin=cur_d.min().min(), vmax=cur_d.max().max(), interpolation='none', extent=extent
    )
    ax.set_title(title)
    ax.set_xlabel('Oscillation \n Bandwidth')
    ax.set_ylabel('Oscillation \n Height')

    return mesh


# %%
f, axs = plt.subplots(ncols=2, figsize=(8, 18))
mesh = plot_mesh_bw_h(df_pivot, 'pyrasa', ax=axs[0])
mesh = plot_mesh_bw_h(df_pivot, 'specparam', ax=axs[1])

f.tight_layout()

cbar = f.colorbar(mesh, ax=axs.ravel().tolist(), orientation='vertical', aspect=4.0, fraction=0.05)
cbar.set_ticks([np.round(df_pivot.min().min(), 2), np.round(df_pivot.max().max(), 2)])
cbar.set_label('Exponent Δ')
f.savefig('../results/delta_exponent_sim_bw_h.svg')

# %%
df_nona = df_p_sim.copy().query('cf < 40').reset_index()
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

ax.set_xlabel('')
sns.despine()

# %%
f, ax = plt.subplots(figsize=(14, 14), ncols=2, nrows=2)
sns.lineplot(df_nona, x='sim_freq', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0, 0])
ax[0, 0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_nona, x='sim_exp', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0, 1])

ax[0, 1].set_xlabel('Simulated Exponent')

sns.lineplot(df_nona, x='sim_bw', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1, 0])

ax[1, 0].set_xlabel('Simulated Oscillation Width')

sns.lineplot(
    df_nona, x='sim_height', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1, 1]
)

ax[1, 1].set_xlabel('Simulated Oscillation Height')

sns.despine()
f.tight_layout()


# %%

df_hp_sp = df_nona.query('method == "specparam"').query('pw > 1').copy()
df_ir = df_p_sim.query('method == "pyrasa"')

df_p_cmb = pd.concat([df_hp_sp, df_ir]).reset_index()

df_p_cmb['Center \n Frequency Δ'] = np.abs(df_p_cmb['cf'] - df_p_cmb['sim_freq'])

f, ax = plt.subplots(figsize=(4, 6))
sns.violinplot(
    df_p_cmb,
    x='method',
    y='Center \n Frequency Δ',
    hue='method',
    # split=True,
    legend=False,
    ax=ax,
)

ax.set_xlabel('')
sns.despine()

# %%
f, ax = plt.subplots(figsize=(14, 14), ncols=2, nrows=2)
sns.lineplot(
    df_p_cmb, x='sim_freq', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0, 0]
)
ax[0, 0].set_xlabel('Simulated Oscillation (Hz)')

sns.lineplot(df_p_cmb, x='sim_exp', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[0, 1])

ax[0, 1].set_xlabel('Simulated Exponent')

sns.lineplot(df_p_cmb, x='sim_bw', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1, 0])

ax[1, 0].set_xlabel('Simulated Oscillation Width')

sns.lineplot(
    df_p_cmb, x='sim_height', y='Center \n Frequency Δ', hue='method', palette=palette, legend=False, ax=ax[1, 1]
)

ax[1, 1].set_xlabel('Simulated Oscillation Height')

sns.despine()
f.tight_layout()

# %%
f, axes = plt.subplots(ncols=3, figsize=(14, 6))

sns.violinplot(
    df_ap_sim,
    x='method',
    y='Exponent Δ',
    hue='method',
    # split=True,
    legend=False,
    ax=axes[0],
)

sns.violinplot(
    df_nona,
    x='method',
    y='Center \n Frequency Δ',
    hue='method',
    # split=True,
    legend=False,
    ax=axes[1],
)

sns.violinplot(
    df_p_cmb,
    x='method',
    y='Center \n Frequency Δ',
    hue='method',
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
