# %%
import joblib
from pathlib import Path
from pyrasa.utils.peak_utils import get_peak_params, get_band_info
from pyrasa.utils.aperiodic_utils import compute_slope
import os
from pymatreader import read_mat
import scipy.signal as dsp

from pyrasa.irasa import irasa

from cluster_jobs.meta_job import Job

# %% load data


class eo_ec_analysis(Job):
    job_data_folder = 'eo_ec'

    def run(
        self,
        subject_id,
        condition,
    ):
        subject_id = int(subject_id)
        base_p = '/'.join(os.path.abspath(os.path.dirname(__file__)).split('/')[:-1])
        if subject_id < 10:
            cur_file = list(
                Path(base_p + '/data/SPIS-Resting-State-Dataset-master/Pre-SART EEG').glob(
                    f'S0{subject_id}*_{condition}.mat'
                )
            )[0]
        else:
            cur_file = list(
                Path(base_p + '/data/SPIS-Resting-State-Dataset-master/Pre-SART EEG').glob(
                    f'S{subject_id}*_{condition}.mat'
                )
            )[0]

        sig = read_mat(cur_file)['dataRest']
        fs = 256

        psd_kwargs = {'nperseg': fs * 4, 'noverlap': fs * 2, 'average': 'median'}

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

        # % compute raw spectrum
        freq, psd = dsp.welch(sig[:64, :], fs=fs, **psd_kwargs)

        # % compute irasa
        freq_irasa, psd_ap, psd_p = irasa(sig, fs=fs, band=(0.1, 30), kwargs_psd=psd_kwargs, hset_info=(1, 2, 0.05))
        # %
        peak_params = get_peak_params(psd_p, freqs=freq_irasa, ch_names=ch_names)
        alpha_df = get_band_info(peak_params, freq_range=(8, 12), ch_names=ch_names)
        alpha_df['s_id'] = subject_id
        alpha_df['condition'] = condition
        # %
        fixed, gof_f = compute_slope(psd_ap, freq_irasa, fit_func='fixed', scale=False, ch_names=ch_names)
        fixed['s_id'] = subject_id
        fixed['condition'] = condition
        gof_f['s_id'] = subject_id
        gof_f['condition'] = condition
        try:
            knee, gof_k = compute_slope(psd_ap, freq_irasa, fit_func='knee', scale=False, ch_names=ch_names)
            knee['s_id'] = subject_id
            knee['condition'] = condition
            gof_k['s_id'] = subject_id
            gof_k['condition'] = condition
        except ValueError:
            knee = None

        data = {
            'periodics': {'peak_params': peak_params, 'alpha': alpha_df},
            'aperiodics': {'fixed': fixed, 'knee': knee},
            'psd': {'psd': psd, 'freq': freq},
        }

        joblib.dump(data, self.full_output_path)
