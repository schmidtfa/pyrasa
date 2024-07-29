import numpy as np
import pytest
from neurodsp.sim import sim_combined, sim_knee, sim_oscillation, sim_powerlaw
from neurodsp.utils.sim import set_random_seed

from .settings import KNEE_FREQ, N_SECONDS

# def pytest_configure(config):
set_random_seed(42)


@pytest.fixture(scope='session')
def combined_signal(exponent, osc_freq, fs):
    components = {'sim_powerlaw': {'exponent': exponent}, 'sim_oscillation': {'freq': osc_freq}}
    yield sim_combined(n_seconds=N_SECONDS, fs=fs, components=components)


@pytest.fixture(scope='session')
def fixed_aperiodic_signal(exponent, fs):
    yield sim_powerlaw(n_seconds=N_SECONDS, fs=fs, exponent=exponent)


@pytest.fixture(scope='session')
def knee_aperiodic_signal(exponent, fs):
    yield sim_knee(n_seconds=N_SECONDS, fs=fs, exponent1=0, exponent2=exponent, knee=KNEE_FREQ ** np.abs(exponent))


@pytest.fixture(scope='session')
def oscillation(osc_freq, fs):
    yield sim_oscillation(n_seconds=N_SECONDS, fs=fs, freq=osc_freq)
