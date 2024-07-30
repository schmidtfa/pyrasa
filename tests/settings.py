# %%

import numpy as np

N_SECONDS = 60
FS = [500, 750, 1000]
OSC_FREQ = [5, 10, 20]
MANY_OSC_FREQ = np.arange(2, 30, 1)
EXPONENT = [-1, -1.5, -2.0]
KNEE_FREQ = 15
EXP_KNEE_COMBO = [  # (1.0, 10.0), (1.0, 15.0), (1., 25.0),
    # TODO: -> figure out why we get systemic knee errors when slope < 1
    # My guess it has to do with this knee ** exp and knee ** (1/exp),
    # where slight values below 1 result in weird knee freqs even if
    # the gof and model looks objectively good (I checked this in my simulations).
    (1.5, 125.0),
    (1.5, 32.0),
    (1.5, 58.0),
    (2.0, 100.0),
    (2.0, 225.0),
    (2.0, 625.0),
]
TOLERANCE = 0.15
KNEE_TOLERANCE = 3
MIN_R2 = 0.8  # seems like a sensible minimum
MIN_R2_SPRINT = 0.7
MIN_CORR_PSD_CMB = 0.99
