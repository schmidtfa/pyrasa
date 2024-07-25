# %%
from pathlib import Path

import numpy as np

N_SECONDS = 90
FS = [500, 750, 1000]
OSC_FREQ = [5, 10, 20]
MANY_OSC_FREQ = np.arange(2, 30, 1)
EXPONENT = [-0.5, -1.0, -2.0]
KNEE_FREQ = [10, 20, 30]
TOLERANCE = 0.1
MIN_R2 = 0.8  # seems like a sensible minimum
MIN_CORR_PSD_CMB = 0.99


# Path Settings
TESTS_PATH = Path(__file__).parent.absolute()
BASE_TEST_FILE_PATH = TESTS_PATH / 'test_files'

# %%
