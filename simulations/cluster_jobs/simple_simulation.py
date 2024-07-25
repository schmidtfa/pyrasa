# %%
from neurodsp.sim import sim_powerlaw, sim_peak_oscillation, sim_combined, sim_knee, sim_bursty_oscillation

from cluster_jobs.meta_job import Job
import joblib
from neurodsp.utils.data import create_times
from neurodsp.sim import set_random_seed

from pyrasa.irasa import irasa, irasa_sprint
from pyrasa.utils.peak_utils import get_peak_params_sprint, get_peak_params
from pyrasa.utils.aperiodic_utils import compute_slope

import scipy.signal as dsp
import numpy as np
import pandas as pd


# %% Collect together parameters for combined signals


class simple_simulation(Job):
    job_data_folder = 'sim_all_simple11'
    exclude_kwargs_from_filename = ['n_secs', 'exp', 'knee', 'freq', 'bw', 'height', 'b_proba']
    include_hash_in_fname = True

    def run(
        self,
        subject_id,
        n_secs=60,
        fs=500,
        exp=-1.0,
        knee_freq=15,
        freq=10,
        bw=1.5,
        height=1.5,
        fit_type='fixed',
        b_proba=0.2,
        min_peak_height=0.01,
        burst_tol_time=0.2,
        n_seeds=30,
    ):
        kind = subject_id
        burst_list, p_ir_l, ap_ir_l = [], [], []
        p_sp_l_f, ap_sp_l_f = [], []
        for seed_val in np.arange(n_seeds):
            kwargs_psd = {'nperseg': fs * 4, 'noverlap': fs * 2}
            set_random_seed(seed_val=seed_val)
            burst_detection = None
            # specparam cannot yet work with double exponents setting exp_1 to 0 for now

            sim_feats = ['sim_exp', 'sim_knee_freq', 'sim_freq', 'sim_bw', 'sim_height', 'sim_b_proba', 'seed', 'kind']
            sim_params = dict(zip(sim_feats, np.zeros(len(sim_feats)) * np.nan))

            if kind == 'exponent_delta':
                sig = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
                sim_params['sim_exp'] = exp
                sim_params['kind'] = kind

            elif kind == 'knee_delta':
                exponent2 = -2
                sig = sim_knee(
                    n_seconds=n_secs, fs=fs, exponent1=0, exponent2=exponent2, knee=knee_freq ** np.abs(exponent2)
                )

                sim_params['sim_exp'] = exponent2
                sim_params['sim_knee_freq'] = knee_freq
                sim_params['kind'] = kind

            elif kind == 'exponent_knee_delta':
                knee_freq = 15
                sig = sim_knee(n_seconds=n_secs, fs=fs, exponent1=0, exponent2=exp, knee=knee_freq ** np.abs(exp))

                sim_params['sim_exp'] = exp
                sim_params['sim_knee_freq'] = knee_freq
                sim_params['kind'] = kind

            elif kind == 'burst_delta':
                # here we want to test irasa and irasa sprint
                # TODO: mark burst times.
                exp = -1.5
                freq = 10
                sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
                sig_bursty = sim_bursty_oscillation(
                    n_seconds=n_secs, fs=fs, freq=freq, burst_params={'enter_burst': b_proba, 'leave_burst': b_proba}
                )
                burst_bool = sig_bursty > 0
                sig = sig_bursty + sig_ap
                sim_params['sim_exp'] = exp
                sim_params['sim_freq'] = freq
                sim_params['sim_b_proba'] = b_proba
                sim_params['kind'] = kind

                # burst detection using pyrasa
                _, periodic, freqs, times = irasa_sprint(
                    sig[np.newaxis, :], fs=fs, band=(1, 100), smooth=False, n_avgs=[3, 5, 7]
                )
                df_peaks = get_peak_params_sprint(periodic, freqs=freqs, times=times, min_peak_height=min_peak_height)

                burst_time = create_times(n_seconds=n_secs, fs=fs)
                burst_ts = burst_time[burst_bool]
                fair_bursts = burst_ts[(burst_ts / 0.02) % 1 == 0]  # need to have similar freq resolution

                detected = []
                for b in fair_bursts:
                    close_list = []
                    for p in df_peaks['time'].to_numpy():
                        close_list.append(np.isclose(p, b, atol=burst_tol_time))
                    detected.append(np.any(close_list))

                burst_detection = {
                    'percent_cycles_detected': np.mean(detected),
                    'n_cycles_detected': df_peaks['time'].to_numpy().shape[0],
                    'n_cycles_simulated': fair_bursts.shape[0],
                    'cycle_detection_ratio': fair_bursts.shape[0] / df_peaks['time'].to_numpy().shape[0],
                }

            elif kind == 'freq_width_delta':
                height = 1.5
                freq = 10
                exp = -1.5
                peak_params = {'freq': freq, 'bw': bw, 'height': height}

                sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
                sig = sim_peak_oscillation(sig_ap, fs=fs, **peak_params)

                sim_params['sim_exp'] = exp
                sim_params['sim_freq'] = freq
                sim_params['sim_bw'] = bw
                sim_params['sim_height'] = height
                sim_params['kind'] = kind

            elif kind == 'freq_height_delta':
                bw = 0.5
                freq = 10
                exp = -1.5
                peak_params = {'freq': freq, 'bw': bw, 'height': height}

                sig_ap = sim_powerlaw(n_seconds=n_secs, fs=fs, exponent=exp)
                sig = sim_peak_oscillation(sig_ap, fs=fs, **peak_params)

                sim_params['sim_exp'] = exp
                sim_params['sim_freq'] = freq
                sim_params['sim_bw'] = bw
                sim_params['sim_height'] = height
                sim_params['kind'] = kind

            elif kind == 'freq_delta':
                comps = {
                    'sim_powerlaw': {
                        'exponent': -1.5,
                    },
                    'sim_oscillation': {'freq': freq},
                }
                sig = sim_combined(n_seconds=n_secs, fs=fs, components=comps)

                sim_params['sim_exp'] = exp
                sim_params['sim_freq'] = freq
                sim_params['sim_bw'] = bw
                sim_params['sim_height'] = height
                sim_params['kind'] = kind

            sim_params['seed'] = seed_val
            df_sim = pd.DataFrame(sim_params, index=[0])

            burst_list.append(burst_detection)

            # do irasa testing
            freqs, aperiodic, periodic = irasa(sig, fs=fs, band=(1, 100), kwargs_psd=kwargs_psd)

            df_ap_f, _ = compute_slope(aperiodic, freqs, scale=False, fit_func=fit_type)
            df_ap_f['method'] = 'pyrasa'
            df_ap_f = pd.concat([df_ap_f, df_sim], axis=1)

            df_p = get_peak_params(periodic, freqs, min_peak_height=min_peak_height, peak_width_limits=(1, 8))
            df_p['method'] = 'pyrasa'
            df_sim_p = pd.DataFrame(np.repeat(df_sim.values, df_p.shape[0], axis=0), columns=df_sim.columns)
            df_p = pd.concat([df_p, df_sim_p], axis=1)

            p_ir_l.append(df_p)
            ap_ir_l.append(df_ap_f)

            # do specparam testing
            from fooof import FOOOF

            fm = FOOOF(
                peak_width_limits=(1, 8),
                max_n_peaks=8,
                min_peak_height=0.1,
                aperiodic_mode=fit_type,
            )

            freqs, psd = dsp.welch(sig, fs=fs, **kwargs_psd)

            df_ap_sp_f, df_p_sp_f = self.extract_specparams(freqs, psd, fm, fit_type=fit_type, df_sim=df_sim)
            p_sp_l_f.append(df_p_sp_f)
            ap_sp_l_f.append(df_ap_sp_f)

            data = {
                'specparam': {
                    'periodic': pd.concat(p_sp_l_f),
                    'aperiodic': pd.concat(ap_sp_l_f),
                },
                'pyrasa': {
                    'periodic': pd.concat(p_ir_l),
                    'aperiodic': pd.concat(ap_ir_l),
                },
                'burst_pyrasa': burst_list,
            }

            joblib.dump(data, self.full_output_path)

    def extract_specparams(self, freqs, psd, fm, fit_type, df_sim):
        # freq_range adjusted to match irasas evaluated frequency range
        # i.e. 2*fmax, fmin/2
        fm.fit(freqs, psd, freq_range=(0.5, 200))
        # aperiodics
        if fit_type == 'fixed':
            df_ap = pd.DataFrame(
                {
                    'Offset': fm.get_params('aperiodic_params')[0],
                    'Exponent': fm.get_params('aperiodic_params')[1],
                },
                index=[0],
            )

        elif fit_type == 'knee':
            df_ap = pd.DataFrame(
                {
                    'Offset': fm.get_params('aperiodic_params')[0],
                    'Knee': fm.get_params('aperiodic_params')[1],
                    'Exponent_2': fm.get_params('aperiodic_params')[2],
                },
                index=[0],
            )

            df_ap['Knee Frequency (Hz)'] = df_ap['Knee'] ** (1.0 / df_ap['Exponent_2'])

        df_ap['fit_type'] = fit_type
        df_ap['method'] = 'specparam'
        # periodics

        peak_params = fm.get_params('peak_params')
        print(peak_params)
        if np.isscalar(peak_params[0]):
            df_p = pd.DataFrame(
                {
                    'cf': np.nan,
                    'pw': np.nan,
                    'bw': np.nan,
                },
                index=[0],
            )
        else:
            df_p = pd.DataFrame(
                peak_params,
                columns=('cf', 'pw', 'bw'),
            )

        df_p['method'] = 'specparam'
        df_sim_p = pd.DataFrame(np.repeat(df_sim.values, df_p.shape[0], axis=0), columns=df_sim.columns)

        df_p = pd.concat([df_p, df_sim_p], axis=1)
        df_ap = pd.concat([df_ap, df_sim], axis=1)

        return df_ap, df_p
