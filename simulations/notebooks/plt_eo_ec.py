# %%
import joblib
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

import mne


# %%
data_l, data_l_ap = [], []
for i in np.arange(2, 11):
    base_p = f'/home/schmidtfa/git/pyrasa/paper/data/eo_ec/{i}/'
    cur_ps = list(Path(base_p).glob('*.dat'))

    cur_data = pd.concat([joblib.load(cur)['aperiodics']['fixed'] for cur in cur_ps])
    data_l_ap.append(cur_data)

    cur_data = pd.concat([joblib.load(cur)['periodics']['alpha'] for cur in cur_ps])
    data_l.append(cur_data)
# %%
all_df_p = pd.concat(data_l)
# %%
df_comp = all_df_p.groupby(['s_id', 'condition'])[['cf', 'bw', 'pw']].mean().reset_index()
# %%
f, axes = plt.subplots(ncols=3, figsize=(12, 4))

swarm_kwargs = {'size': 10, 'alpha': 0.5}
point_kwargs = {'markers': '_', 'markersize': 20}

pretty_labels = ['Center Frequency (Hz)', 'Bandwidth (Hz)', 'Power']

for ix, (cond, gl) in enumerate(zip(['cf', 'bw', 'pw'], pretty_labels)):
    sns.stripplot(data=df_comp, x='condition', y=cond, hue='condition', ax=axes[ix], **swarm_kwargs)
    sns.pointplot(data=df_comp, x='condition', y=cond, hue='condition', ax=axes[ix], **point_kwargs)
    axes[ix].set_xlabel('')
    axes[ix].set_ylabel(gl)

sns.despine()
f.tight_layout()

f.savefig('../results/periodic_eo_ec.svg')

# %%
all_df_ap = pd.concat(data_l_ap)
all_df_ap.drop(columns='fit_type', inplace=True)
# %%
df_comp = all_df_ap.groupby(['s_id', 'condition'])[['Offset', 'Exponent']].mean().reset_index()
# %%
f, axes = plt.subplots(ncols=2, figsize=(8, 4))

swarm_kwargs = {'size': 10, 'alpha': 0.5}
point_kwargs = {'markers': '_', 'markersize': 20}

for ix, cond in enumerate(['Offset', 'Exponent']):
    sns.stripplot(data=df_comp, x='condition', y=cond, hue='condition', ax=axes[ix], **swarm_kwargs)
    sns.pointplot(data=df_comp, x='condition', y=cond, hue='condition', ax=axes[ix], **point_kwargs)
    axes[ix].set_xlabel('')

sns.despine()
f.tight_layout()

f.savefig('../results/aperiodic_eo_ec.svg')
# %%

df_cmb = (
    all_df_p.merge(all_df_ap, on=['s_id', 'condition', 'ch_name'])
    .groupby(['s_id', 'condition'])[['Offset', 'Exponent', 'cf', 'bw', 'pw']]
    .mean()
    .reset_index()
)
corrs = df_cmb.drop(columns='s_id').groupby('condition').corr('spearman')

ec_corr = corrs.loc['EC']
eo_corr = corrs.loc['EO']

# %%
f, ax = plt.subplots(figsize=(16, 6), ncols=2)
mask = np.triu(np.ones_like(ec_corr, dtype=bool))
sns.heatmap(ec_corr, annot=True, vmin=-1, vmax=1, cmap='RdBu_r', ax=ax[0])
sns.heatmap(eo_corr, annot=True, vmin=-1, vmax=1, cmap='RdBu_r', ax=ax[1])

# %%
df_cmb.query('condition == "EC"')[['Offset', 'Exponent', 'cf', 'bw', 'pw']].corr()
# %%
ch_names = [
    'Fp1',
    'AF7',
    'AF3',
    'F1',
    'F3',
    'F5',
    'F7',
    'FT7',
    'FC5',
    'FC3',
    'FC1',
    'C1',
    'C3',
    'C5',
    'T7',
    'TP7',
    'CP5',
    'CP3',
    'CP1',
    'P1',
    'P3',
    'P5',
    'P7',
    'P9',
    'PO7',
    'PO3',
    'O1',
    'Iz',
    'Oz',
    'POz',
    'Pz',
    'CPz',
    'Fpz',
    'Fp2',
    'AF8',
    'AF4',
    'Afz',
    'Fz',
    'F2',
    'F4',
    'F6',
    'F8',
    'FT8',
    'FC6',
    'FC4',
    'FC2',
    'FCz',
    'Cz',
    'C2',
    'C4',
    'C6',
    'T8',
    'TP8',
    'CP6',
    'CP4',
    'CP2',
    'P2',
    'P4',
    'P6',
    'P8',
    'P10',
    'PO8',
    'PO4',
    'O26',
]

# %%
df_conds = all_df_p.groupby(['condition', 'ch_name']).median().reset_index()
ec = df_conds.query('condition == "EC"')
eo = df_conds.query('condition == "EO"')

ec = ec.set_index('ch_name').reindex(ch_names)
eo = eo.set_index('ch_name').reindex(ch_names)
# %%
montage = mne.channels.make_standard_montage(kind='biosemi64')
info = mne.create_info(ch_names=montage.ch_names, sfreq=264, ch_types='eeg')

eo_im = mne.EvokedArray(eo['pw'].to_numpy()[:, np.newaxis], info=info)
eo_im.set_montage('biosemi64')

ec_im = mne.EvokedArray(ec['pw'].to_numpy()[:, np.newaxis], info=info)
ec_im.set_montage('biosemi64')

ec_eo_im = mne.EvokedArray((ec['pw'].to_numpy() - eo['pw'].to_numpy())[:, np.newaxis], info=info)
ec_eo_im.set_montage('biosemi64')

# %%
min, max = eo['pw'].to_numpy().mean(), ec['pw'].to_numpy().max()

sns.set_context('talk')
f, ax = plt.subplots(figsize=(4, 4), nrows=2)
eo_im.plot_topomap(
    ch_type='eeg',
    times=0,
    colorbar=False,  # vlim=(min, max),
    cmap='Reds',
    axes=ax[0],
)

ec_im.plot_topomap(ch_type='eeg', times=0, colorbar=False, vlim=(min, max), cmap='Reds', axes=ax[1])
# %%
f, ax = plt.subplots(figsize=(4, 4))
sns.set_context('talk')
ec_eo_im.plot_topomap(ch_type='eeg', times=0, colorbar=False, cmap='Reds', axes=ax)
# %%
