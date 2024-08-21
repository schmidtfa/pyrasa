# %%

import numpy as np

N_SECONDS = 60
FS = [500, 750, 1000]

OSC_FREQ = [5, 10, 20]
MANY_OSC_FREQ = np.arange(2, 30, 1)
# through a bug in neurodsp #223 there is a systematic error in generating spectra with high frequency peaks
# where the actual simulated center frequency is biased.
# This bias increases as the simulated center frequency gets closer to the sampling rate used in the simulation.
# If you want to play around with this check: simulations/notebooks/sim_peak_tests.py
EXPONENT = [-1, -1.5, -2.0]
KNEE_FREQ = 15
KNEE_FREQS = [10, 15, 25]
# There seems to be a higher error in knee fits for knee and exponent estimates when
# the difference in pre and post knee exponent is low. This kinda makes sense
# TODO: Test this systematically -> see whether this is an issue with irasa or slope fitting in general
EXP_KNEE_COMBO = [
    (-1.0, 10.0),
    (-1.0, 15.0),
    (-1.0, 25.0),
    (-1.5, 125.0),
    (-1.5, 32.0),
    (-1.5, 58.0),
    (-2.0, 100.0),
    (-2.0, 225.0),
    (-2.0, 625.0),
]  # we test exp + knee combined as both relate to each other
TOLERANCE = 0.3  # 0.15
SPRINT_TOLERANCE = 0.4
HIGH_TOLERANCE = 0.5

KNEE_TOLERANCE = 5
MIN_R2 = 0.8  # seems like a sensible minimum
MIN_R2_SPRINT = 0.7
MIN_CORR_PSD_CMB = 0.99
