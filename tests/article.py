from sigpipes.physionet import PhysionetRecord
from sigpipes.sigoperator import Sample, Tee
from sigpipes.psigoperator import ParTee
from sigpipes.plotting import Plot
from sigpipes.sigfilter import Butter, FiltFilt, MedFilt
from functools import partial

plot = partial(Plot, annot_specs=[("atr/N", "atr/A")] * 2, dir="/tmp")

ekg = PhysionetRecord("100", "mitdb").sigcontainer(["atr"]) | Sample(3.0, 6.5) | plot(file="signal")
(ekg | plot(file="signal")
     | ParTee(
            FiltFilt(Butter(3, 10.0)) | plot(file="signal_butter"),
            MedFilt(9) | plot(file="signal_medfilt")
        )
)
