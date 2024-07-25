# %%
from neurodsp.sim import sim_powerlaw, sim_peak_oscillation, sim_combined, sim_knee

import joblib

import scipy.signal as dsp
import numpy as np
import pandas as pd
import os
import scipy.stats as sts

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

new_rc_params = {'text.usetex': False, 'svg.fonttype': 'none'}
mpl.rcParams.update(new_rc_params)

sns.set_style('ticks')
sns.set_context('poster')

pal = [  #'#cccccc',
    '#969696',
    '#636363',
    '#252525',
]


n_secs = 60
fs = 500
kwargs_psd = {'nperseg': fs * 4, 'noverlap': fs * 2}

from pathlib import Path

base_p = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1]) + '/data/sim_all_simple11/'


def get_simulations(folder):
    all_files = list(Path(base_p + folder).glob('*.dat'))

    ap_list, p_list = [], []
    for f in all_files:
        cur_f = joblib.load(f)

        cur_p = pd.concat([cur_f['specparam']['periodic'], cur_f['pyrasa']['periodic']])  # .dropna(axis=1)

        cur_ap = pd.concat([cur_f['specparam']['aperiodic'], cur_f['pyrasa']['aperiodic']])  # .dropna(axis=1)

        ap_list.append(cur_ap)
        p_list.append(cur_p)

    df_ap = pd.concat(ap_list)
    df_p = pd.concat(p_list)

    return df_ap, df_p


# %% Setting #1 - varying slope
exp_list = [-1.0, -1.5, -2]

kwargs_psd = {'nperseg': fs * 4, 'noverlap': fs * 2}
sigs = []
for exp in exp_list:
    sig = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['exponent'] = np.abs(exp)
    sigs.append(df)

df_1 = pd.concat(sigs)


df_ap, df_p = get_simulations('exponent_delta')
# df_p = df_p.query('cf < 40').reset_index()
f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 100'),
    legend=False,
    x='Frequency (Hz)',
    y='Power',
    hue='exponent',
    palette=pal,
    ax=ax[0],
)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[1].scatter(
    df_ap.query('method == "pyrasa"')['Exponent'],
    df_ap.query('method == "specparam"')['Exponent'],
)
ax[1].set_xlabel('Exponent \n (PyRASA)')
ax[1].set_ylabel('Exponent \n (Specparam)')
ax[1].set_xlim(0, 3)
ax[1].set_ylim(0, 3)

r, p = sts.pearsonr(
    df_ap.query('method == "pyrasa"')['Exponent'],
    df_ap.query('method == "specparam"')['Exponent'],
)
r, p = np.round(r, 2), np.round(p, 3)
ax[1].annotate(f'r = {r}', xy=(1.5, 0.6))
ax[1].annotate(f'p = {p}', xy=(1.5, 0.2))

n_runs = len(df_p['seed'].unique()) * len(df_p['sim_exp'].unique())
df_p = df_p.query('cf < 40')
df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()
f.tight_layout()

f.savefig('../results/exponent_delta_sim.svg')

# %% Setting #2 - varying knee
knee_list = [10.0, 20.0, 40.0]

sigs = []
for knee in knee_list:
    exp = -2
    sig = sim_knee(n_seconds=n_secs, fs=fs, exponent1=0, exponent2=exp, knee=knee ** np.abs(exp))
    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['knee'] = knee
    sigs.append(df)

df_1 = pd.concat(sigs)


df_ap, df_p = get_simulations('knee_delta')

# %%
f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 100'), legend=False, x='Frequency (Hz)', y='Power', hue='knee', palette=pal, ax=ax[0]
)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

ax[1].scatter(
    df_ap.query('method == "pyrasa"')['Knee Frequency (Hz)'],
    df_ap.query('method == "specparam"')['Knee Frequency (Hz)'],
)
ax[1].set_xlabel('Knee Frequency \n (Hz; PyRASA)')
ax[1].set_ylabel('Knee Frequency \n (Hz; Specparam)')
ax[1].set_xlim(0, 40)
ax[1].set_ylim(0, 40)


r, p = sts.pearsonr(
    df_ap.query('method == "pyrasa"')['Knee Frequency (Hz)'],
    df_ap.query('method == "specparam"')['Knee Frequency (Hz)'],
)
r, p = np.round(r, 2), np.round(p, 3)
ax[1].annotate(f'r = {r}', xy=(20, 8))
ax[1].annotate(f'p = {p}', xy=(20, 2))

f.tight_layout()
sns.despine()


n_runs = len(df_p['seed'].unique()) * len(df_p['sim_knee_freq'].unique())
df_p = df_p.query('cf < 40')
df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()
f.tight_layout()

f.savefig('../results/knee_delta_sim.svg')

# %% Setting #3 - varying exponent post knee
exp_list = [-1.0, -2.0, -3.0]

sigs = []
for exp in exp_list:
    sig = sim_knee(n_seconds=n_secs, fs=fs, exponent1=0, exponent2=exp, knee=15 ** np.abs(exp))
    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['exp'] = exp
    sigs.append(df)

df_1 = pd.concat(sigs)


# %%
f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 100'), legend=False, x='Frequency (Hz)', y='Power', hue='exp', palette=pal, ax=ax[0]
)
ax[0].set_xscale('log')
ax[0].set_yscale('log')

df_ap, df_p = get_simulations('exponent_knee_delta')

ax[1].scatter(
    df_ap.query('method == "pyrasa"')['Exponent_2'],
    df_ap.query('method == "specparam"')['Exponent_2'],
)
ax[1].set_xlabel('Exponent \n (PyRASA)')
ax[1].set_ylabel('Exponent \n (Specparam)')
ax[1].set_xlim(1, 4)
ax[1].set_ylim(1, 4)

f.tight_layout()

r, p = sts.pearsonr(
    df_ap.query('method == "pyrasa"')['Exponent_2'],
    df_ap.query('method == "specparam"')['Exponent_2'],
)

r, p = np.round(r, 2), np.round(p, 3)
ax[1].annotate(f'r = {r}', xy=(2.5, 1.5))
ax[1].annotate(f'p = {p}', xy=(2.5, 1.1))


n_runs = len(df_p['seed'].unique()) * len(df_p['sim_knee_freq'].unique())
df_p = df_p.query('cf < 40')
df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()
f.tight_layout()

f.savefig('../results/exponent_knee_delta_sim.svg')

sns.despine()
# %% Setting #4 - freq_delta
sigs = []
osc_list = [10, 15, 20]
for freq in osc_list:
    comps = {
        'sim_powerlaw': {
            'exponent': -1.5,
        },
        'sim_oscillation': {'freq': freq},
    }
    sig = sim_combined(n_seconds=n_secs, fs=fs, components=comps)
    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['osc'] = freq
    sigs.append(df)

df_1 = pd.concat(sigs)


f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 50').query('`Frequency (Hz)` > 1'),
    legend=False,
    x='Frequency (Hz)',
    y='Power',
    hue='osc',
    palette=pal,
    ax=ax[0],
)

ax[0].set_xscale('log')
ax[0].set_yscale('log')

df_ap, df_p = get_simulations('freq_delta')

n_runs = len(df_p['seed'].unique()) * len(df_p['sim_freq'].unique())
df_p = df_p.query('cf < 40')
df_p['Center Frequency Δ'] = np.abs(df_p['cf'] - df_p['sim_freq'])

sns.barplot(
    df_p,  # legend=False,
    x='method',
    y='Center Frequency Δ',
    hue='method',
    palette='deep',
    hue_order=['specparam', 'pyrasa'],
    ax=ax[1],
)
ax[1].set_xlabel('')


df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()
f.tight_layout()

f.savefig('../results/freq_delta_sim.svg')

# %% Setting #5 - freq_width_delta
sigs = []
bw_list = [1, 2, 3]
for bw in bw_list:
    height = 1.5
    freq = 10
    exp = -1.5
    peak_params = {'freq': freq, 'bw': bw, 'height': height}

    sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
    sig = sim_peak_oscillation(sig_ap, fs=fs, **peak_params)

    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['bw'] = bw
    sigs.append(df)

df_1 = pd.concat(sigs)

f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 50').query('`Frequency (Hz)` > 2'),
    legend=False,
    x='Frequency (Hz)',
    y='Power',
    hue='bw',
    palette=pal,
    ax=ax[0],
)

ax[0].set_xscale('log')
ax[0].set_yscale('log')


df_ap, df_p = get_simulations('freq_width_delta')
df_p = df_p.reset_index()
n_runs = len(df_p['seed'].unique()) * len(df_p['sim_freq'].unique())
df_p = df_p.query('cf < 40')
df_p['Bandwidth Δ'] = np.abs(df_p['bw'] - df_p['sim_bw'])

sns.barplot(
    df_p,  # legend=False,
    x='method',
    y='Bandwidth Δ',
    hue='method',
    palette='deep',
    ax=ax[1],
)
ax[1].set_xlabel('')

df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()
f.tight_layout()

f.savefig('../results/bw_delta_sim.svg')

# %% Setting #5 - height_delta
sigs = []
h_list = [1, 1.5, 2]
for height in h_list:
    bw = 0.5
    freq = 10
    exp = -1.5
    peak_params = {'freq': freq, 'bw': bw, 'height': height}

    sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
    sig = sim_peak_oscillation(sig_ap, fs=fs, **peak_params)

    f, psd = dsp.welch(sig, fs=fs, **kwargs_psd)
    df = pd.DataFrame({'Frequency (Hz)': f, 'Power': psd})
    df['height'] = height
    sigs.append(df)

df_1 = pd.concat(sigs)

# %%
f, ax = plt.subplots(figsize=(15, 5), ncols=3)

sns.lineplot(
    df_1.query('`Frequency (Hz)` < 50').query('`Frequency (Hz)` > 2'),
    legend=False,
    x='Frequency (Hz)',
    y='Power',
    hue='height',
    palette=pal,
    ax=ax[0],
)

ax[0].set_xscale('log')
ax[0].set_yscale('log')

df_ap, df_p = get_simulations('freq_height_delta')
df_ap = df_ap.reset_index()
df_ap['Exponent Δ'] = np.abs(df_ap['Exponent'] - np.abs(df_ap['sim_exp']))

n_runs = len(df_p['seed'].unique()) * len(df_p['sim_freq'].unique())
df_p = df_p.query('cf < 40').query('cf > 4').reset_index()
df_p.loc[df_p['method'] == 'pyrasa', 'pw'] = (df_p.loc[df_p['method'] == 'pyrasa', 'pw']) * 45
df_p['Power Δ'] = np.abs((df_p['sim_height'] - df_p['pw']))


sns.lineplot(df_p, legend=False, x='sim_height', y='pw', hue='method', palette='deep', ax=ax[1])
ax[1].set_xlabel('Simulated Peak Height')
ax[1].set_ylabel('Peak Height')

df_p2 = df_p[['cf', 'pw', 'bw', 'method']].dropna(axis=0)
osc_per_run = pd.DataFrame(df_p2.groupby('method').apply(len) / n_runs).reset_index()
osc_per_run.columns = ['method', 'Oscillations detected \n (per Model fit)']
sns.barplot(
    osc_per_run,
    x='method',
    y='Oscillations detected \n (per Model fit)',
    hue='method',
    hue_order=['specparam', 'pyrasa'],
    palette='deep',
    ax=ax[2],
)
ax[2].set_xlabel('')
sns.despine()

f.tight_layout()

f.savefig('../results/pw_delta_sim.svg')

# %%


# %%
plt.scatter(
    df_p.query('method == "specparam"')['pw'],  # *15,
    df_p.query('method == "specparam"')['sim_height'],
)


# %%
plt.scatter(
    df_p.query('method == "pyrasa"')['pw'],  # *15,
    df_p.query('method == "pyrasa"')['sim_height'],
)

# %%
sts.pearsonr(
    df_p.query('method == "pyrasa"').dropna(axis=1)['pw'],  # *15,
    df_p.query('method == "pyrasa"').dropna(axis=1)['sim_height'],
)


# %%
import scipy.signal as dsp

test_arr = np.array(
    [
        1,
        1,
        1,
        2,
        3,
        4,
        5,
        4,
        3,
        2,
        1,
        1,
    ]
)
dsp.find_peaks(test_arr, height=[0, 5], prominence=0.5)
# %%
height = 1
freq = 10
exp = -1.0
bw = 0.5
peak_params = {'freq': freq, 'bw': bw, 'height': height}
n_secs = 60
fs = 500
sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
sig = sim_peak_oscillation(sig_ap, fs=fs, **peak_params)

# %%
freq, psd = dsp.welch(sig, fs=fs, nperseg=fs * 2, noverlap=fs)
psd.max()
# %%
plt.loglog(freq, psd)
# %%
dsp.find_peaks(
    psd,
    height=[0, 5],
    prominence=0.01,
)
# %%
from pyrasa import irasa
from pyrasa.utils.peak_utils import get_peak_params

# %%
freq, ap, p = irasa(sig, fs=fs, band=(1, 100), kwargs_psd={'nperseg': fs * 2, 'noverlap': fs})
# %%
plt.plot(freq, p[0, :])
# %%
dsp.find_peaks(p[0, :], height=[0, 1], prominence=0.05)
# %%
get_peak_params(p, freq)
# %%
