from sigpipes.sigcontainer import SigContainer
import numpy as np
from sigpipes.physionet import PhysionetRecord
from sigpipes.megawin import MegaWinMatlab

from random import sample
from sigpipes.sigoperator import *
import pytest


@pytest.fixture(scope="module", params=["testdata/chuze A.mat", "testdata/chuze B.mat",
                                        "testdata/test.mat"])
def megawindata(request):
    mat = MegaWinMatlab(request.param)
    return mat.sigcontainer()


@pytest.fixture(scope="module", params=[("records/chuzeA", None)])
def physiodata(request):
    p = PhysionetRecord(*request.param)
    return p.sigcontainer()


@pytest.fixture(scope="module", params=[("records/chuzeA", None)])
def markeddata(request):
    p = PhysionetRecord(*request.param)
    return p.sigcontainer(["markers"])


@pytest.fixture(scope="module")
def simpledata():
    sig = np.array(
        [
            [0, 1, 0, -1, 0, 1, 0, -1, 0],
            [1, 1, 1,  1, 1, 1, 1,  1, 1]
        ])
    c = SigContainer.from_signal_array(signals=sig,
                                       channels=["saw", "ones"], units=["mV", "mV"], fs=1)
    return c


def test_simpledata_load(simpledata):
    assert simpledata.sample_count == 9
    assert simpledata.channel_count == 2


def test_physiodata_load(physiodata):
    assert physiodata.channel_count == len(physiodata["signals/channels"])


def test_megawindata_load(megawindata):
    assert megawindata.channel_count == len(megawindata["signals/channels"])


def test_hdf5_roundtrip(megawindata):
    original = megawindata | Hdf5("/tmp/test.h5")
    serialized = SigContainer.from_hdf5("/tmp/test.h5")
    assert np.all(original.signals == serialized.signals)


def test_ufunc(megawindata):
    basket = {}
    (megawindata | UfuncOnSignals(np.abs) | Tee( UfuncOnSignals(np.negative) | Reaper(basket, "negative"))
                                          | Reaper(basket, "positive"))
    psig = basket["positive"].signals
    nsig = basket["negative"].signals
    assert np.sum(psig) == - np.sum(nsig)


def test_annot_partitioner(markeddata):
    pars = markeddata | MarkerSplitter("markers", left_outer_segments=True,
                                       right_outer_segment=True)

    assert markeddata.sample_count == sum(par.sample_count for par in pars)
    for par in pars:
        for annot in par["annotations"].values():
            assert all(0 <= p < par.sample_count for p  in annot["samples"])


def test_partitioner(megawindata):
    size = megawindata.sample_count
    frag_count = 5
    points = sample(range(size), frag_count)
    if 0 not in points:
        points.append(0)
        frag_count += 1
    if size not in points:
        points.append(size)
        frag_count += 1
    fragments = megawindata | SampleSplitter(points)
    assert len(fragments) == frag_count - 1
    assert megawindata.sample_count == sum(frag.sample_count for frag in fragments)


def test_features(simpledata):
    r1, r2 = (simpledata | AltOptional(UfuncOnSignals(np.vectorize(lambda x: x + 1)))
                         | FeatureExtraction(wamp_threshold=0.5))
    assert r1["/meta/features/MAV"][0] == 4/9
    assert r1["/meta/features/MAV"][1] == 1
    assert r2["/meta/features/MAV"][0] == 1
    assert r2["/meta/features/MAV"][1] == 2

    assert r1["/meta/features/VAR"][0] == 4/8
    assert r1["/meta/features/VAR"][1] == 9/8
    assert r2["/meta/features/VAR"][0] == 13/8
    assert r2["/meta/features/VAR"][1] == 36/8

    assert r1["/meta/features/WL"][0] == 8
    assert r1["/meta/features/WL"][1] == 0
    assert r2["/meta/features/WL"][0] == 8
    assert r2["/meta/features/WL"][1] == 0

    assert r1["/meta/features/WAMP"]["0.5"][0] == 8
    assert r1["/meta/features/WAMP"]["0.5"][1] == 0
    assert r2["/meta/features/WAMP"]["0.5"][0] == 8
    assert r2["/meta/features/WAMP"]["0.5"][1] == 0

    assert r1["/meta/features/MMAV1"][0] == 3/9
    assert r1["/meta/features/MMAV1"][1] == 7/9
    assert r2["/meta/features/MMAV1"][0] == 7/9
    assert r2["/meta/features/MMAV1"][1] == 14/9


def test_convolution(simpledata):
    c = simpledata | Convolution([1, 1, 1])
    assert c.sample_count == simpledata.sample_count
    sig = simpledata.signals
    nsig = c.signals
    assert np.all((sig[:, 5]+sig[:, 6]+sig[:, 7])/3.0 == nsig[:, 6])