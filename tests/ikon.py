import matplotlib
from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import (
    Print, UfuncOnSignals, Convolution, FeatureExtraction,
    SampleSplitter)
from sigpipes.joiner import JoinChannels
from sigpipes.plotting import Plot, GraphOpts
import numpy as np
from glob import glob


for filename in glob("/mnt/windows/Dokumenty/26_02_2020_11_38/1hp.txt"):
    print(filename)
    (SigContainer.from_csv(filename, header=False, fs=250,
                           annotation="/home/fiser/IKON/ICON/anotace/1hp.a.csv")
     | Print()
     | Plot(annot_specs=[("1hp.a/S", "1hp.a/H")], graph_opts=GraphOpts(title=[filename]))).show()
