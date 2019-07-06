from sigpipes.pandas import *
import pytest
from sigpipes.megawin import MegaWinMatlab
import numpy as np


@pytest.fixture(scope="module", params=["testdata/chuze A.mat", "testdata/chuze B.mat",
                                        "testdata/test.mat"])
def megawindata(request):
    mat = MegaWinMatlab(request.param)
    return mat.sigcontainer()


def test_pandas(megawindata):
    df = megawindata | DataFrame()
    for i in range(df.channel_count):
        assert np.all(df.loc[:, megawindata["signals/channels"][i]]
                                == megawindata["signals/data"][i])
