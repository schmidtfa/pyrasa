from neurodsp.sim import sim_combined

from pyrasa import irasa, irasa_sprint
from pyrasa.tests.conftest import combined_signal
from pyrasa.tests.test_settings import FS

def test_irasa(combined_signal):

    # Estimate periodic and aperiodic components with IRASA
    f_range = [1, 30]
    freqs, psd_ap, psd_pe = irasa(combined_signal, FS, f_range, noverlap=int(2*FS))
    assert len(freqs) == len(psd_ap) == len(psd_pe)