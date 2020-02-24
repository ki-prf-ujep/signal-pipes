import numpy as np
from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import SigOperator, Alternatives
from warnings import warn

from typing import Sequence
from scipy.signal import correlate

class Joiner(SigOperator):
    def __init__(self, *branches):
        self.subop = branches

    def fromSources(self) -> SigContainer:
        container = self.prepare_container(self.subop[0])
        return self.join(container, self.subop[1:])

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        containers = [inp if isinstance(inp, SigContainer) else container | inp for inp in self.subop]
        # containers = container | Alternatives(*self.subop)
        assert all(isinstance(c, SigContainer) for c in containers), "Join over non containers"
        self.assertion(container, containers)
        return self.join(container, containers)

    def prepare_container(self, container: SigContainer) -> SigContainer:
        return SigContainer(container.d.deepcopy(shared_folders=["annotations"],
                                                 empty_folders=["meta"]))

    def join(self, output: SigContainer, inputs: Sequence[SigContainer]) -> SigContainer:
        raise NotImplementedError("abstract method")

    def assertion(self, output: SigContainer, inputs: Sequence[SigContainer]) -> None:
        assert all(c.signals.shape == output.signals.shape for c in inputs)
        if any(c.d["signals/fs"] != output.d["signals/fs"] for c in inputs):
            warn("Join operation on signals with incompatible frequencies")

class Sum(Joiner):
    def __init__(self, *branches) -> None:
        super().__init__(*branches)

    def join(self, outcontainer: SigContainer, incontainers: Sequence[SigContainer]) -> SigContainer:
        result = np.zeros_like(incontainers[0].signals)
        for inc in incontainers:
            result += inc.signals
        outcontainer.d["signals/data"] = result
        return outcontainer

class JoinChannels(Joiner):
    def __init__(self, *branches):
        super().__init__(*branches)

    def join(self, output: SigContainer, inputs: Sequence[SigContainer]) -> SigContainer:
        for input in inputs:
            output.d["signals/data"] = np.vstack((output.signals, input.signals))
            output.d["signals/channels"].extend(input.d["signals/channels"])
            output.d["signals/units"].extend(input.d["signals/units"])
        return output

    def assertion(self, output: SigContainer, inputs: Sequence[SigContainer]) -> None:
        assert all(c.signals.shape[1] == output.signals.shape[1] for c in inputs)
        if any(c.d["signals/fs"] != output.d["signals/fs"] for c in inputs):
            warn("Join operation on signals with incompatible frequencies")


class Concatenate(Joiner):
    def __init__(self, *branches, channel_names = None):
        super().__init__(*branches)
        self.names = channel_names

    def join(self, output: SigContainer, inputs: Sequence[SigContainer]) -> SigContainer:
        if self.names is not None:
            output.d["signals/channels"] = self.names
        for input in inputs:
            output.d["signals/data"] = np.hstack((output.signals, input.signals))
            if self.names is None:
                output.d["signals/channels"] = [
                    output.d["signals/channels"][i] + " + " + input.d["signals/channels"][i]
                        for i in range(len(output.d["signals/channels"]))]
        return output

    def assertion(self, output: SigContainer, inputs: Sequence[SigContainer]) -> None:
        assert all(c.signals.shape[0] == output.signals.shape[0] for c in inputs)
        if any(c.d["signals/fs"] != output.d["signals/fs"] for c in inputs):
            warn("Join operation on signals with incompatible frequencies")


class CrossCorrelate(Joiner):
    def __init__(self, *branches, mode:str = 'full', method: str = 'auto'):
        super().__init__(*branches)
        self.mode = mode
        self.method = method

    def assertion(self, output: SigContainer, inputs: Sequence[SigContainer]) -> None:
        assert all(c.signals.shape[0] == output.signals.shape[0] for c in inputs)

    def join(self, output: SigContainer, inputs: Sequence[SigContainer]) -> SigContainer:
        assert len(inputs)<=1, "Cross corelation with more than two signal is not supported"
        in1 = output
        in2 = inputs[0] if inputs else in1
        result = correlate(in1.signals, in2.signals, self.mode, self.method)
