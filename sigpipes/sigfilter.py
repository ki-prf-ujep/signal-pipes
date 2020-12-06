from numbers import Real

import numpy as np
from sigpipes.sigoperator import SigModifierOperator
from sigpipes.sigcontainer import SigContainer
import scipy.signal as sig
from typing import Any, Union, Tuple


class Butter:
    def __init__(self, order: int, cutoff: Union[Real, Tuple[Real, Real]], btype: str):
        self.order = order
        self.cutoff = cutoff
        self.btype = btype

    def __call__(self, fs: float):
        sos = sig.butter(self.order, self.cutoff, btype=self.btype, analog=False, output="sos", fs=fs)
        return sos

    def __str__(self):
        if self.btype == "lowpass":
            band = f"<{self.cutoff}"
        elif self.btype == "highpass":
            band = f">{self.cutoff}"
        elif self.btype == "bandpass":
            band = f"[{self.cutoff[0]}-{self.cutoff[1]}]"
        elif self.btype == "bandstop":
            band = f"]{self.cutoff[0]}-{self.cutoff[1]}["
        return f"butter{band}"


class Filter(SigModifierOperator):
    def __init__(self, coeff_generator, **params) -> None:
        self.cg = coeff_generator
        self.params = params

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        fs = container.d["signals/fs"]
        sos = self.cg(fs)
        container.d["signals/data"] = sig.sosfilt(sos, container.d["signals/data"], **self.params)
        return container

    def log(self):
        return f"{str(self.cg)}"


class MedFilt(SigModifierOperator):
    def __init__(self, window_length: int = 3) -> None:
        self.window_length = window_length

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        result = np.empty_like(container.d["signals/data"])
        for i in range(container.channel_count):
            result[i] = sig.medfilt(container.d["signals/data"][i, :], self.window_length)
        container.d["signals/data"] = result
        return container

    def log(self):
        return f"MM_{self.window_length}"


class Hilbert(SigModifierOperator):
    def apply(self, container: SigContainer) -> Any:
        container = self.prepare_container(container)
        h = sig.hilbert(container.signals, axis=1)
        container.d["signals/data"] = np.sqrt(np.real(h)**2 + np.imag(h)**2)
        return container
