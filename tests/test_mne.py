import pytest
from neurodsp.utils.sim import set_random_seed

from pyrasa.irasa_mne import irasa_epochs, irasa_raw

set_random_seed(42)


@pytest.mark.filterwarnings('ignore:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:UserWarning')
def test_mne(gen_mne_data_raw):
    mne_data, epochs = gen_mne_data_raw

    # test raw
    aperiodic_mne, periodic_mne = irasa_raw(
        mne_data, band=(0.25, 50), duration=2, hset_info=(1.0, 2.0, 0.05), as_array=False
    )
    aperiodic_mne.get_slopes(fit_func='fixed')
    periodic_mne.get_peaks(smoothing_window=2)

    # test epochs
    aperiodic, periodic = irasa_epochs(epochs, band=(0.5, 50), hset_info=(1.0, 2.0, 0.05), as_array=False)
    aperiodic.get_slopes(fit_func='fixed', scale=True)
    periodic.get_peaks(smoothing_window=2)
