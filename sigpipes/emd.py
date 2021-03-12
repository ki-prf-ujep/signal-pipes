from PyEMD import EMD
import numpy as np

from sigpipes.sigoperator import SigModifierOperator
from sigpipes.sigcontainer import SigContainer

class Emd(SigModifierOperator):
    def __init__(self, include_source=True):
        self.include = include_source

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        assert container.channel_count == 1, "only one channel signal is supported"
        emd = EMD()
        IMFs = emd(container.signals[0,:])
        oname = container.d["signals/channels"][0]
        if self.include:
            container.d["signals/data"] = np.vstack((container.signals, IMFs))
            container.d["signals/channels"] = [oname] + [f"{oname} IMF_{i+1}" for i in range(IMFs.shape[0])]
            container.d["signals/units"] *= IMFs.shape[0] + 1
        else:
            container.d["signals/data"] = IMFs
            container.d["signals/channels"] = [f"{oname} IMF_{i+1}" for i in range(IMFs.shape[0])]
            container.d["signals/units"] *= IMFs.shape[0]
        return container

    def log(self):
            return "EMD"
