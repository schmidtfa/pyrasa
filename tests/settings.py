# %%

import numpy as np

N_SECONDS = 60
FS = [500, 750, 1000]
OSC_FREQ = [5, 10, 20]
MANY_OSC_FREQ = np.arange(2, 30, 1)
EXPONENT = [-0.5, -1.0, -2.0]
KNEE_FREQ = 15
TOLERANCE = 0.2
KNEE_TOLERANCE = 3
MIN_R2 = 0.8  # seems like a sensible minimum
MIN_CORR_PSD_CMB = 0.99
