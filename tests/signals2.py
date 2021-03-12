from glob import iglob
from re import compile

from sigpipes.auxtools import resuffix
from sigpipes.joiner import CrossCorrelate
from sigpipes.plotting import Plot, FftPlot
from sigpipes.sigfilter import Filter, Butter
from sigpipes.sigoperator import Print, Sample, Csv, Fft, PResample, AltOptional, Alternatives, ChannelSelect, \
    MVNormalization
from sigpipes.sources import SynergyLP
from sigpipes.sigcontainer import SigContainer
from sigpipes.emd import Emd

sources = "/home/fiser/data/emg2/*.txt"
for file in sorted(iglob(sources)):
    print(file)
    signal = SigContainer.hdf5_cache(SynergyLP(file, shortname=compile("^[^-]+"), channels=["needle", "surface"]),
                                     Csv() | PResample(down=10) | Csv())
    (signal
        | Sample(20.0, 20.5)
        | Filter(Butter(5, 20, btype="highpass"))
        | Alternatives(ChannelSelect([0]), ChannelSelect([1]))
        | Emd()
        | Plot()
        | Fft()
        | FftPlot())
    continue

    (signal | AltOptional(Filter(Butter(5, 20, btype="highpass")))
            | Fft()
            | Plot()
            | FftPlot(frange = (10, 1000))
            | Alternatives(Sample(20.0, 30.0), Sample(20.0, 20.3)) | Plot()
    )
    fsig = signal | Filter(Butter(5, 20, btype="highpass"))
    (fsig | ChannelSelect([0, 0, 1, 1]) | CrossCorrelate(fsig | ChannelSelect([0, 1, 0, 1]))
          | Sample(0.0, 20.0)
          | MVNormalization()
          | Plot()
    )

