from sigpipes.sources import SynergyLP
from sigpipes.sigoperator import Print, PResample, Sample
from sigpipes.plotting import Plot, GraphOpts
from sigpipes.sigoperator import CSVSaver
from glob import glob

#path = "../tests/testdata/emg.data"
#outdir = "/home/fiser/Dokumenty/outputs"

#signals = [Synergy(path, sec).sigcontainer() for sec in Synergy.section_iter(path)]
#signals = [Synergy(path, sec).sigcontainer() for sec in Synergy.section_iter(path)]
#JoinChannels(*signals).fromSources() \
#    | CSVSaver("synergyV.csv", dir=outdir) \
#    | Plot(graph_opts=GraphOpts(columns=5), file="graphsV.png", dir=outdir)

#Concatenate(*signals, channel_names=["Concatenated signal"]).fromSources() \
#    | CSVSaver("synergyH.csv", dir=outdir) | Plot(file="graphsH.png", dir=outdir)


#SigContainer.from_csv("synergyV.csv", dir=outdir) | Print()


for file in glob("/home/fiser/data/2020-25-02-SB/*.txt"):
    signal = SynergyLP(file).sigcontainer()
    (signal | PResample(50) | Print() | Plot()).savefig(file+".png")

    signal = SynergyLP(file).sigcontainer()
    (signal | Sample(6.0, 6.1) | Print() | Plot()).savefig(file+"_detail.png")