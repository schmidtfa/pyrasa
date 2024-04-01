#%%
import sys
#from neurodsp.sim import sim_combined
from neurodsp.sim import sim_knee #,sim_powerlaw, sim_oscillation, sim_variable_oscillation, sim_damped_oscillation

from cluster_jobs.meta_job import Job
import numpy as np
import scipy.signal as dsp
import joblib
import pandas as pd


sys.path.append('../../')

from pyrasa.irasa import irasa
from pyrasa.aperiodic_utils import compute_slope


#%%
# Simulation settings
# Set some general settings, to be used across all simulations

class knee_simulations(Job):
    
    job_data_folder = 'knee_sim'
    
    def run(self,
            fs=1000,
            n_seconds=60,
            duration=4,
            hset=(1.,2.,.01),
            fmax_rasa=250,
            knee_freq=20,
            fit_func='knee',
            exponent_1=0,
            ):
            
        #needs to be twice of fmax to match the evaluated frequency range from irasa
        fmax_plot = fmax_rasa * np.max(hset) 
        nfft = 2**(np.ceil(np.log2(int(fs*duration*np.max(hset)))))
        # welch settings
        kwargs_psd = {'window': 'hann',
                    'nperseg': int(fs*duration), 
                    'nfft': nfft,
                    'noverlap': 0}


        aperiodic_fits, gofs = [], []
        exponents = np.arange(0.5, 6, step=0.5)

        for exponent in exponents:
            cur_signal = sim_knee(n_seconds, 
                                    fs, 
                                    exponent1=-1*exponent_1, 
                                    exponent2=-1*exponent, 
                                    knee=knee_freq ** (exponent_1 + exponent))
                

            # Calculate original spectrum
            freq_ap, psds_ap = dsp.welch(cur_signal, fs=fs, **kwargs_psd)
            freq_rasa_ap, psd_aperiodics_ap, _ = irasa(cur_signal, 
                                                        band=(0.1, fmax_rasa), #calculate fmin aswell
                                                        fs=fs, kwargs_psd=kwargs_psd,
                                                        hset_info=hset)


            freq_mask = np.logical_and(freq_ap < fmax_plot,  freq_ap > 0)

            def _add_info(df, param_type, knee_freq, exponent):

                df['param_type'] = param_type
                df['GT_Exponent'] = exponent
                df['GT_Knee_Freq'] = knee_freq
                df['GT_Knee'] = knee_freq ** (exponent_1 + exponent)

                return df

            # now lets parametrize the fractal part (and compare to ground truth)
            ap_rasa, gof_rasa = compute_slope(freq_rasa_ap,  psd_aperiodics_ap[0,:], fit_func=fit_func)
            ap_rasa = _add_info(ap_rasa, 'irasa', knee_freq, exponent)
            gof_rasa = _add_info(gof_rasa, 'irasa', knee_freq, exponent)
            # also evaluate slope fit on aperiodic spectra without IRASA
            ap_norasa, gof_norasa = compute_slope(freq_ap[freq_mask],  psds_ap[freq_mask], fit_func=fit_func)
            ap_norasa = _add_info(ap_norasa, 'norasa', knee_freq, exponent)
            gof_norasa = _add_info(gof_norasa, 'norasa', knee_freq, exponent)

            df_cmb = pd.concat([ap_rasa, ap_norasa])
            aperiodic_fits.append(df_cmb)

            df_gof_cmb = pd.concat([gof_rasa, gof_norasa])
            gofs.append(df_gof_cmb)


        # %% save
        data = {'gof': pd.concat(gofs),
                'ap_fits': aperiodic_fits}

        joblib.dump(data, self.full_output_path)