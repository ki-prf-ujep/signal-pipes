from sigpipes.physionet import PhysionetRecord
from sigpipes.sigoperator import Sample, Identity, Scale, AltOptional, Hdf5, MarkerSplitter, FeatureExtraction
from sigpipes.psigoperator import ParTee
from sigpipes.joiner import Sum
from sigpipes.plotting import Plot
from sigpipes.sigfilter import Butter, FiltFilt, MedFilt, Hilbert
from functools import partial

# preparation
plot = partial(Plot, annot_specs=[("atr/N", "atr/A")] * 2, dir="/tmp")
hdf5 = partial(Hdf5, dir="/tmp")

filter = Sum(Identity(),
             MedFilt(101) | MedFilt(201) | Scale(-1.0)
) | FiltFilt(Butter(5, 10.0))

# signal preprocessing
ekg = PhysionetRecord("100", "mitdb").sigcontainer(["atr"]) | Sample(3.0, 6.5)
signal = (
    ekg | filter  # application of pre-created filter
        | AltOptional(Hilbert()) # fork = one branch with Hilbert second without
            | MarkerSplitter("atr")  # branch for each fragment between annot. markers
                # saves, plots features for each fragment and alternative
                | hdf5("/tmp/{0.id}.h5")  # six files saved
                | plot(file="{0.id}.png") # six plots saved
)



