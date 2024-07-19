import os
import shutil

import pytest
from neurodsp.sim import sim_combined
from neurodsp.utils.sim import set_random_seed

from pyrasa.tests.test_settings import BASE_TEST_FILE_PATH, EXPONENT, FREQ, FS, N_SECONDS


def pytest_configure(config):
    set_random_seed(42)


@pytest.fixture(scope='session')
def combined_signal():
    components = {'sim_powerlaw': {'exponent': EXPONENT}, 'sim_oscillation': {'freq': FREQ}}
    yield sim_combined(n_seconds=N_SECONDS, fs=FS, components=components)


@pytest.fixture(scope='session', autouse=True)
def check_dir():
    """Once, prior to session, this will clear and re-initialize the test file directories."""

    # If the directories already exist, clear them
    if os.path.exists(BASE_TEST_FILE_PATH):
        shutil.rmtree(BASE_TEST_FILE_PATH)

    # Remake (empty) directories
    os.mkdir(BASE_TEST_FILE_PATH)
