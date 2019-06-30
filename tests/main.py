from sigpipes.megawin import MegaWinMatlab
from sigpipes.physionet import PhysionetRecord
from sigpipes.sigoperator import *
from sigpipes.plotting import *
from sigpipes.pandas import DataFrame

MegaWinMatlab("testdata/chuze A.mat").write_to("chuzeA", "records")
p = PhysionetRecord("records/chuzeA")

(p.sigcontainer(["markers"])
 | ChannelSelect([0, 1, 2])
 | Tee(Sample(0.0, 30.0)
        | PResample(new_freq=10)
        | Hdf5("/tmp/posun.h5")
        | Plot(annot_specs=[None, "markers"], file="/tmp/{0.id}.svg"))
 | Sample(0.0, np.timedelta64(1, "m"))
 | AltOptional(UfuncOnSignals(np.abs))
 | AltOptional(Convolution(np.ones(999)))
 | Plot(annot_specs=["markers"], file="/tmp/{0.id}.svg")
 | FeatureExtraction(wamp_threshold=(0.01, 0.02))
 | Fft()
 | FftPlot(file="/tmp/fft@{0.id}.svg")
 | Print()
 | MarkerSplitter("markers", left_outer_segments=True)
   | FeatureExtraction()
   | Plot(file="/tmp/{0.id}.svg")
   | Hdf5("/tmp/{0.id}.h5")
   | DataFrame("/tmp/{0.id}.data")
 )