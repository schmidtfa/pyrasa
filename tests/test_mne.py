import pytest
from neurodsp.utils.sim import set_random_seed
from scipy.optimize import OptimizeWarning

from pyrasa.irasa_mne import irasa_epochs, irasa_raw

set_random_seed(42)


@pytest.mark.filterwarnings('ignore:RuntimeWarning')
@pytest.mark.filterwarnings('ignore:UserWarning')
def test_mne(gen_mne_data_raw):
    mne_data, epochs = gen_mne_data_raw

    # test raw
    irasa_raw_result = irasa_raw(mne_data, band=(0.25, 50), duration=2, hset_info=(1.0, 2.0, 0.05))
    with pytest.warns(OptimizeWarning):
        irasa_raw_result.aperiodic.fit_aperiodic_model(fit_func='fixed')
        irasa_raw_result.periodic.get_peaks(smoothing_window=2)

    # test epochs
    irasa_epoched_result = irasa_epochs(epochs, band=(0.5, 50), hset_info=(1.0, 2.0, 0.05))
    irasa_epoched_result.aperiodic.fit_aperiodic_model(fit_func='fixed', scale=True)
    irasa_epoched_result.periodic.get_peaks(smoothing_window=2)
