from neurodsp.utils.sim import set_random_seed

from pyrasa.irasa_mne import irasa_epochs, irasa_raw

set_random_seed(42)


def test_mne(gen_mne_data_raw):
    mne_data, epochs = gen_mne_data_raw

    # test raw
    irasa_raw_result = irasa_raw(mne_data, band=(0.25, 50), duration=2, hset_info=(1.0, 2.0, 0.05))

    irasa_raw_result.fit_aperiodic_model(fit_func='fixed')
    irasa_raw_result.get_peaks(smoothing_window=2)

    # Joint channel selection (in place)
    irasa_raw_result.pick(['MEG 0111', 'MEG 0121'])

    # Joint channel removal and reordering
    irasa_raw_result.drop_channels('MEG 0121')

    # Independent copy
    irasa_raw_result.copy()

    # test epochs
    irasa_epoched_result = irasa_epochs(epochs, band=(0.5, 50), hset_info=(1.0, 2.0, 0.05))
    irasa_epoched_result.fit_aperiodic_model(fit_func='fixed', scale=True)
    irasa_epoched_result.get_peaks(smoothing_window=2)

    # Joint channel selection (in place)
    irasa_epoched_result.pick(['MEG 0111', 'MEG 0121'])

    # Joint channel removal and reordering
    irasa_epoched_result.drop_channels('MEG 0121')

    # Independent copy
    irasa_epoched_result.copy()
