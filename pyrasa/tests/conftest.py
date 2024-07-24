import shutil

import numpy as np
import pytest
from neurodsp.sim import sim_combined, sim_knee, sim_powerlaw
from neurodsp.utils.sim import set_random_seed

from pyrasa.tests.test_settings import BASE_TEST_FILE_PATH, N_SECONDS


def pytest_configure(config):
    set_random_seed(42)


@pytest.fixture(scope='session')
def combined_signal(exponent, osc_freq, fs):
    components = {'sim_powerlaw': {'exponent': exponent}, 'sim_oscillation': {'freq': osc_freq}}
    yield sim_combined(n_seconds=N_SECONDS, fs=fs, components=components)


@pytest.fixture(scope='session')
def fixed_aperiodic_signal(exponent, fs):
    yield sim_powerlaw(n_seconds=N_SECONDS, fs=fs, exponent=exponent)


@pytest.fixture(scope='session')
def knee_aperiodic_signal(exponent, knee_freq, fs):
    yield sim_knee(n_seconds=N_SECONDS, fs=fs, exponent1=0, exponent2=exponent, knee=knee_freq ** np.abs(exponent))


@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if BASE_TEST_FILE_PATH.exists():
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    BASE_TEST_FILE_PATH.mkdir(parents=True, exist_ok=True)
