#%%
from neurodsp.sim import sim_combined, sim_knee
import numpy as np
from os.path import join

n_secs=60
fs_list=[500, 750, 1000]
exp1=0
exp2_list=[-1., -1.5, -2.0]
knee_freq_list=[10, 15, 25]
osc_list = [5, 10, 20]

base_folder = '/Users/fabian.schmidt/git/pyrasa/tests/test_data/'

for fs in fs_list:
    for exp2 in exp2_list:
        for osc_freq in osc_list:
            for knee_freq in knee_freq_list:
                #% generate and save knee osc
                knee = knee_freq ** np.abs(exp2)
                components = {'sim_knee': {'exponent1': exp1, 'exponent2': exp2, 'knee': knee}, 
                            'sim_oscillation': {'freq': osc_freq}
                            }
                cmb_sim = sim_combined(n_seconds=n_secs, fs=fs, components=components)

                fname = f'cmb_sim__fs_{fs}__exp1_{np.abs(exp1)}__exp2_{np.abs(exp2)}_knee_{np.round(knee, 0)}__osc_freq_{osc_freq}_.npy'
                np.save(join(base_folder + 'knee_osc_data', fname), cmb_sim, allow_pickle=False)

                #% generate and save knee
                knee_sim = sim_knee(n_seconds=n_secs, fs=fs, exponent1=exp1, exponent2=exp2, knee=knee)

                fname = f'knee_sim__fs_{fs}__exp1_{np.abs(exp1)}__exp2_{np.abs(exp2)}_knee_{np.round(knee, 0)}_.npy'
                np.save(join(base_folder + 'knee_data', fname), knee_sim, allow_pickle=False)

# %%
