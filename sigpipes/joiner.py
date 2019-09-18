import numpy as np
from sigpipes.sigcontainer import SigContainer
from sigpipes.sigoperator import SigOperator, Alternatives

from typing import Sequence, Union, Iterable, Optional, MutableMapping, Any


class Joiner(SigOperator):
    def __init__(self, *branches):
        self.subop = branches

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        containers = container | Alternatives(*self.subop)
        assert all(isinstance(c, SigContainer) for c in containers), "Join over non containers"
        assert all(c.signals.shape == container.signals.shape for c in containers)
        assert all(c.d["signals/fs"] == container.d["signals/fs"] for c in containers)
        return self.join(container, containers)

    def prepare_container(self, container: SigContainer) -> SigContainer:
        return SigContainer(container.d.deepcopy(shared_folders=["annotations"],
                                                 empty_folders=["meta"]))

    def join(self, output: SigContainer, inputs: Sequence[SigContainer]) -> SigContainer:
        raise NotImplementedError("abstract method")


class Sum(Joiner):
    def __init__(self, *branches) -> None:
        super().__init__(*branches)

    def join(self, outcontainer: SigContainer, incontainers: Sequence[SigContainer]) -> SigContainer:
        result = np.zeros_like(incontainers[0].signals)
        for inc in incontainers:
            result += inc.signals
        outcontainer.d["signals/data"] = result
        return outcontainer

