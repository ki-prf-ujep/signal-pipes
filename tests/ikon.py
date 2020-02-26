#import matplotlib
#matplotlib.use("TkAgg")
from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import (
    Print, UfuncOnSignals, Convolution, FeatureExtraction,
    SampleSplitter)
from sigpipes.joiner import JoinChannels
from sigpipes.plotting import Plot, GraphOpts
import numpy as np
from glob import glob


for filename in glob("/mnt/windows/Dokumenty/ikon_coolest_data/[12]*.txt"):
    print(filename)
    (SigContainer.from_csv(filename, header=False, fs=250, annotation="test.csv")
     | Plot(annot_specs=[("test/S", "test/H", "test/F")], graph_opts=GraphOpts(title=[filename]))).show()

