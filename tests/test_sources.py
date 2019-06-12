from sigpipes.sigcontainer import SigContainer
import numpy as np
from sigpipes.physionet import PhysionetRecord


def test_from_data():
    sig = np.ones((2, 100))
    c = SigContainer.from_signal_array(signals=sig,
                                       channels=["ch1", "ch2"],
                                       units=["mV", "mV"])
    assert c.sample_count == 100
    assert c.channel_count == 2


def test_from_physionet():
    p = PhysionetRecord("records/chuzeA")
    c = p.sigcontainer()
    assert c.sample_count == 185115
