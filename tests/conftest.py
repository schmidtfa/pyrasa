import numpy as np
import pytest
from neurodsp.sim import sim_combined, sim_knee, sim_oscillation, sim_powerlaw
from neurodsp.utils.sim import set_random_seed

from .settings import N_SECONDS

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
def knee_aperiodic_signal(exponent, fs, knee_freq):
    yield sim_knee(n_seconds=N_SECONDS, fs=fs, exponent1=0, exponent2=exponent, knee=knee_freq ** np.abs(exponent))


@pytest.fixture(scope='session')
def oscillation(osc_freq, fs):
    yield sim_oscillation(n_seconds=N_SECONDS, fs=fs, freq=osc_freq)


@pytest.fixture(scope='session')
def ts4sprint():
    fs = 500
    alpha = sim_oscillation(n_seconds=0.5, fs=fs, freq=10)
    no_alpha = np.zeros(len(alpha))
    beta = sim_oscillation(n_seconds=0.5, fs=fs, freq=25)
    no_beta = np.zeros(len(beta))

    exp_1 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-1)
    exp_2 = sim_powerlaw(n_seconds=2.5, fs=fs, exponent=-2)

    # %%
    alphas = np.concatenate([no_alpha, alpha, no_alpha, alpha, no_alpha])
    betas = np.concatenate([beta, no_beta, beta, no_beta, beta])

    sim_ts = np.concatenate(
        [
            exp_1 + alphas,
            exp_1 + alphas + betas,
            exp_1 + betas,
            exp_2 + alphas,
            exp_2 + alphas + betas,
            exp_2 + betas,
        ]
    )
    yield sim_ts
