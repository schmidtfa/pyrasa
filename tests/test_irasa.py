#%%
import sys
from neurodsp.sim import sim_combined
from neurodsp.sim import sim_knee, sim_powerlaw, sim_oscillation
from neurodsp.utils import create_times
import numpy as np
import scipy.signal as dsp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('ticks')
sns.set_context('poster')
sys.path.append('../')

from irasa import irasa


#%%
# Simulation settings
# Set some general settings, to be used across all simulations
fs = 500
n_seconds = 60
duration=2
overlap=0.5

# Create a times vector for the simulations
times = create_times(n_seconds, fs)

#%%
exponents = [1, 2, 3]
knee_freq = 15
#%
knee_ap = []
knee_osc = []
for exponent in exponents:

    knee_ap.append(sim_knee(n_seconds, fs, 
                        exponent1=-0.5, 
                        exponent2=-1*exponent, 
                        knee=knee_freq ** exponent))
    
    knee_osc.append(sim_combined(n_seconds, 
                                fs, 
                                components={'sim_oscillation': {'freq' : 10},
                                            'sim_knee': {'exponent1': -0.5, 
                                                        'exponent2':-1*exponent, 
                                                        'knee': knee_freq ** exponent}}))

    
# %% plot timeseries

tmax = times < 1
fig, ax = plt.subplots(figsize=(6,3))

ax.plot(times[tmax], knee_ap[0][tmax], label=f'slope = -{exponents[0]}')
ax.plot(times[tmax], knee_ap[1][tmax]+3, label=f'slope = -{exponents[1]}')
ax.plot(times[tmax], knee_ap[2][tmax]+6, label=f'slope = -{exponents[2]}')

sns.despine()
plt.legend()

# %% test irasa
kwargs_psd = {'window': 'hann',
              'nperseg': int(fs*duration), 
              'noverlap': int(fs*duration*overlap)}

# Calculate original spectrum
freq, psds = dsp.welch(knee_ap, fs=fs, **kwargs_psd)

psd_aperiodics, psd_periodics, freq_rasa = irasa(np.array(knee_ap), band=(0.1, 100), duration=duration, fs=fs)


freq_mask = freq < 100

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for ix, ax in enumerate(axes):
    ax.loglog(freq[freq_mask], psds[ix, freq_mask])
    ax.loglog(freq_rasa, psd_aperiodics[ix])

#%% now do the same with oscillations
from peak_utils import get_peak_params

freq, psds = dsp.welch(knee_osc, fs=fs, **kwargs_psd)
psd_aperiodics, psd_periodics, freq_rasa = irasa(np.array(knee_osc), band=(0.1, 100), duration=duration, fs=fs)

freq_mask = freq < 100

fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

for ix, ax in enumerate(axes):
    ax.loglog(freq[freq_mask], psds[ix, freq_mask])
    ax.loglog(freq_rasa, psd_aperiodics[ix])

get_peak_params(psd_periodics, 
                freq_rasa,
                min_peak_height=0.01,
                peak_width_limits=(0.5, 8),
                peak_threshold=1)

# %%
plt.plot(psd_periodics[0,:])
plt.plot(psd_periodics[1,:])
plt.plot(psd_periodics[2,:])
# %% now lets parametrize the fractal part

from patsy import dmatrix

#%%

#1. Transform data to log log
#2. Specify expected maximum number of knees
#3. loop over sensible cut point specifications based on quantiles of the data -> compute mse in each go
#4. compare mse using bic along with the fitted knees

x = freq_rasa
y = psd_aperiodics[2]
n_knees = 1

x_log,  y_log = np.log10(x), np.log10(y)

knots = tuple(np.round(np.quantile(x_log, np.linspace(0, 1, n_knees)), 4))

spline_matrix = dmatrix(f"bs(aperiodic, knots={knots}, degree=2, include_intercept=True)",
                        {"aperiodic": x_log},return_type='matrix') # matrix or dataframe


#%%
import statsmodels.api as sm

#%%
#from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
from scipy.optimize import curve_fit

def fit(x, a0, a1):
    res = a0*x + a1*x
    return [res]


#curve_fit(fit, spline_matrix, y_log)
#linregress(x=spline_matrix.to_numpy(), y=y_log)


#%%
spline_matrix
cs = sm.GLM(y_log, np.asarray(spline_matrix)).fit()
#array([0.15418033, 0.6909883, 0.15418033])

#%%
plt.plot(x_log, cs.predict())
plt.plot(x_log, y_log)


# %%
cs.summary()


#%%
xp = np.linspace(x_log.min(),x_log.max(), 100)
pred = cs.predict(dmatrix(f"bs(xp, knots={knots}, include_intercept=True)", 
                          {"xp": xp}, 
                          return_type='dataframe'))


# %%
