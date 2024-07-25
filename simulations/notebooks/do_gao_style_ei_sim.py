# %%
from neurodsp.sim import sim_synaptic_current
from neurodsp.utils import create_times
import pandas as pd

import scipy.signal as dsp
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')


# %% based on info from gao 2017)
def make_gao_ei(n_secs, fs, ei_ratio=4):
    """Do lfp gao style parameters as in Gao et al. 2017"""

    mv_rest = -0.65
    mv_rev_e, mv_rev_i = 0, -0.80
    ampa = sim_synaptic_current(n_secs, fs=fs, n_neurons=8000, firing_rate=2, tau_r=0.0001, tau_d=0.002, t_ker=1) * (
        mv_rest - mv_rev_e
    )
    gaba = sim_synaptic_current(n_secs, fs=fs, n_neurons=2000, firing_rate=5, tau_r=0.0005, tau_d=0.01, t_ker=1) * (
        mv_rest - mv_rev_i
    )

    lfp = ampa * ei_ratio + gaba

    return lfp


nsecs = 30
fs = 1000
times = create_times(nsecs, fs)
# %%
ratios = 1 / np.arange(1, 6)
lfps = [make_gao_ei(n_secs=nsecs, fs=fs, ei_ratio=i) for i in ratios]
freqs, psds = zip(*([dsp.welch(lfp, fs=fs, nperseg=4 * fs, noverlap=2 * fs) for lfp in lfps]))

# %%
cmap = sns.color_palette('Greys', np.shape(lfps)[0])
f, ax = plt.subplots(figsize=(4, 4))
for ix, psd in enumerate(psds):
    ax.loglog(freqs[0], psd, color=cmap[ix])

# %%
from fooof import FOOOFGroup

fg = FOOOFGroup(max_n_peaks=0, aperiodic_mode='knee')
fg.fit(freqs[0], np.array(psds))

df = pd.DataFrame(fg.get_params('aperiodic_params'), columns=['offset', 'knee', 'exponent'])

df['Knee Frequency'] = df['knee'] ** (1.0 / df['exponent'])
df['tau'] = 1.0 / (2 * np.pi * df['Knee Frequency'])
# df['ei'] = ['1/2', '1/3', '1/4', '1/5', '1/6']
df['ei'] = 1 / ratios
# %%
plt.scatter(df['ei'], df['Knee Frequency'])
# %%
df.corr('spearman')
# %%
df
# %%
