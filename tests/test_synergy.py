from sigpipes.sources import SynergyLP
from sigpipes.sigoperator import Print, Sample, FeatureExtractor, ChannelSelect, Fft, MVNormalization, \
    RangeNormalization, FFtAsSignal
from sigpipes.plotting import Plot, FftPlot, GraphOpts
from sigpipes.sigoperator import CSVSaver, Hdf5
from glob import iglob
from pathlib import Path
from sigpipes.joiner import JoinChannels, CrossCorrelate
from sigpipes.pandas_support import FeatureFrame
import numpy as np

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

gopts = GraphOpts(sharey=True)
(JoinChannels(*signals).fromSources() | CSVSaver(rename(sources, "signals.csv"))
                                      | Hdf5(rename(sources, "signals.hdf5"))
                                      | Plot(file=rename(sources, "signals_full.png"), graph_opts=gopts)
)

eqsig = (JoinChannels(*signals).fromSources() | ChannelSelect([0, 1, 2])
                                      | Sample(4.0, 8.0)
                                      | MVNormalization()
                                      | CSVSaver(rename(sources, "processed_normalized_part.csv"))
                                      | Hdf5(rename(sources, "processed_normalized_part.hdf5"))
                                      | Plot(file=rename(sources,"signals_part.png"), graph_opts=gopts)
                                      | FeatureExtractor(dict(IEMG=True,
                                                              MAV=True,
                                                              SSI=True,
                                                              VAR=True,
                                                              RMS=True,
                                                              WL=True,
                                                              SC=[0.0001, 0.0002]))
                                      | Sample(5.0, 5.2)
                                      | Plot(file=rename(sources, "signals_detail.png"), graph_opts=gopts)
)

# (eqsig | FeatureFrame()).to_excel(rename(sources,"eqsignals.xls"))

eqsig | Fft() | FftPlot(file=rename(sources,"signals_part_spectrum.png"),
                        frange=(1, 1200)) | FFtAsSignal() | CSVSaver(rename(sources, "signals_part_spectre.csv"))

(eqsig | ChannelSelect([0, 0, 0, 1, 1, 2]) | CrossCorrelate(eqsig | ChannelSelect([0, 1, 2, 1, 2, 2]), mode="full")
         | Sample(0, 1.5)
         | RangeNormalization(-1, 1)
         | CSVSaver(rename(sources, "signals_part_correlations.csv"))
         | Plot(file=rename(sources, "signals_part_correlation.png"))
)
