from sigpipes.sources import SynergyLP
from sigpipes.sigoperator import Print, PResample, Sample, FeatureExtractor, ChannelSelect, Fft
from sigpipes.plotting import Plot, FftPlot
from sigpipes.sigoperator import CSVSaver, Hdf5
from sigpipes.sigcontainer import SigContainer
from glob import iglob
from pathlib import Path
from sigpipes.joiner import JoinChannels, CrossCorrelate
from sigpipes.pandas_support import FeatureFrame
import numpy as np

def resuffix(filename, newsuffix, extension=""):
    parts = list(Path(filename).parts)
    stem = Path(parts[-1]).stem
    parts[-1] = stem + extension + "." + newsuffix
    return Path(*parts).absolute()

def rename(path, newname):
    parts = list(Path(path).parts)
    parts[-1] = newname
    return Path(*parts).absolute()

sources = "/home/fiser/data/emg/*.txt"
signals = [SynergyLP(file).sigcontainer() | Sample(0, 450_000) for file in sorted(iglob(sources))]
#fs = 50000
#xdata = 2 * np.pi * np.arange(0, 450_000) / fs
#ydata = np.sin(xdata*20)+ 0.3*np.sin(xdata*50) + 0.5*np.sin(xdata*5)
# signals.append(SigContainer.from_signal_array(ydata, ["test signal"], ["u"], fs))
(JoinChannels(*signals).fromSources() | CSVSaver(rename(sources, "signals.csv"))
                                      | Hdf5(rename(sources, "signals.hdf5"))
                                      | Plot(file=rename(sources,"signals.png"))
)

eqsig = (JoinChannels(*signals).fromSources() | ChannelSelect([1,2])
                                      | Sample(4.0, 8.0)
                                      | Plot(file=rename(sources,"eqsignals.png"))
                                      | FeatureExtractor(dict(IEMG=True,
                                                              MAV=True,
                                                              SSI=True,
                                                              VAR=True,
                                                              RMS=True,
                                                              WL=True,
                                                              SC=[0.0001, 0.0002]))
)

(eqsig | FeatureFrame()).to_excel(rename(sources,"eqsignals.xls"))

eqsig | Fft() | FftPlot(file=rename(sources,"eqsignals_spectrum.png"), frange=(1,120))

eqsig | CrossCorrelate() | Sample(-1.5, 1.5) | Plot(file=rename(sources, "auto_correlation.png"))

(eqsig | ChannelSelect([0]) | CrossCorrelate(eqsig | ChannelSelect([1]), mode="full")
         | Print()
         | Plot(file=rename(sources, "correlation.png"))
)
