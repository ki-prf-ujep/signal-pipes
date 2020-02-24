from sigpipes.sources import Synergy
from sigpipes.sigoperator import Print
from sigpipes.plotting import Plot, GraphOpts
from sigpipes.joiner import JoinChannels, Concatenate
from sigpipes.sigoperator import CSVSaver
from sigpipes.sigcontainer import SigContainer

path = "../tests/testdata/emg.data"
outdir = "/home/fiser/Dokumenty/outputs"

signals = [Synergy(path, sec).sigcontainer() for sec in Synergy.section_iter(path)]
#signals = [Synergy(path, sec).sigcontainer() for sec in Synergy.section_iter(path)]
JoinChannels(*signals).fromSources() \
    | CSVSaver("synergyV.csv", dir=outdir) \
    | Plot(graph_opts=GraphOpts(columns=5), file="graphsV.png", dir=outdir)

Concatenate(*signals, channel_names=["Concatenated signal"]).fromSources() \
    | CSVSaver("synergyH.csv", dir=outdir) | Plot(file="graphsH.png", dir=outdir)


SigContainer.from_csv("synergyV.csv", dir=outdir) | Print()