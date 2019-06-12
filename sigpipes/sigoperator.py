from sigpipes.sigcontainer import SigContainer, TimeUnit
from sigpipes.auxtools import seq_wrap, smart_tostring

from typing import Sequence, Union, Iterable, Optional
import collections.abc
import sys
import fractions

import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft


class SigOperator:
    """
    Base abstract class of signal operators.
    """
    def apply(self, container: SigContainer) :
        raise NotImplemented("Abstract method")

    def __ror__(self, container: Union[SigContainer, Iterable[SigContainer], "SigOperator"]
                ) -> Union[SigContainer, Iterable[SigContainer], "SigOperator"]:
        if isinstance(container, SigContainer):
            container.d["log"].append(self.log())
            return self.apply(container)
        elif isinstance(container, collections.abc.Iterable):
            return [c | self for c in container]
        elif isinstance(container, SigOperator):
            return CompoundSigOperator(container, self)

    def log(self):
        return "#" + self.__class__.__name__


class CompoundSigOperator(SigOperator):
    def __init__(self, left_operator: SigOperator, right_operator: SigOperator) -> None:
        self.left = left_operator
        self.right = right_operator

    def apply(self, container: SigContainer):
        return container | self.left | self.right


class Print(SigOperator):
    def __init__(self, output=sys.stdout, header=True):
        if isinstance(output, str):
            self.output = open(output, "wt")
        else:
            self.output = output
        self.header = header

    def apply(self, container: SigContainer) -> SigContainer:
        if self.header:
            print(container.id, file=self.output)
            print("-"*40, file=self.output)
        print(str(container), file=self.output)
        return container


class PartitionerTool:
    @staticmethod
    def time_unit_mapper(value):
        if isinstance(value, int):
            return TimeUnit.SAMPLE
        elif isinstance(value, float):
            return TimeUnit.SECOND
        elif isinstance(value, np.timedelta64):
            return TimeUnit.TIME_DELTA
        else:
            raise TypeError("The time unit can not be infered or it is ambiguous")

    @staticmethod
    def to_sample(shift, fs, time_unit):
        if time_unit == TimeUnit.SAMPLE:
            return shift
        elif time_unit == TimeUnit.SECOND:
            return int(shift * fs)
        elif time_unit == TimeUnit.TIME_DELTA:
            interval = int(1_000_000_000 / fs)
            return int(shift / np.timedelta64(interval, "ns"))


class SubSample(SigOperator):
    def __init__(self, start: Union[int, float, np.timedelta64],
                 end: Union[int, float, np.timedelta64]):
        self.start = start
        self.end = end

    def apply(self, container: SigContainer) -> SigContainer:
        fs = container.d["signals/fs"]
        start = PartitionerTool.to_sample(self.start, fs,
                                          PartitionerTool.time_unit_mapper(self.start))
        end = PartitionerTool.to_sample(self.end, fs,
                                        PartitionerTool.time_unit_mapper(self.end))
        container.d["signals/data"] = container.d["signals/data"][:, start:end]

        if "annotations" in container.d:
            adict = container.d["annotations"]
            newdict = SigContainer.cut_annots(adict, start, end)
            adict.update(newdict)

        return container

    def log(self):
        return f"samp({str(self.start)}, {str(self.end)})"


class ChannelSelect(SigOperator):
    def __init__(self, selector: Sequence[int]) -> None:
        self.selector = selector

    def apply(self, container: SigContainer) -> SigContainer:
        container.d["signals/data"] = container.d["signals/data"][self.selector, :]
        container.d["signals/channels"] = np.array(container.d["signals/channels"])[self.selector]
        container.d["signals/units"] = np.array(container.d["signals/units"])[self.selector]
        return container

    def log(self):
        return f"chsel({self.selector})"


class FeatureExtraction(SigOperator):
    def __init__(self, target: str = "features", wamp_threshold: Union[float, Sequence[float]] = ()):
        self.wamp_threshold = seq_wrap(wamp_threshold)
        self.target = target

    def apply(self, container: SigContainer) -> SigContainer:
        n = container.sample_count
        data = container.d["signals/data"]
        absum = np.sum(np.abs(data), axis=1)
        container.d[f"meta/{self.target}/IEMG"] = absum
        container.d[f"meta/{self.target}/MAV"] = absum / n

        data1 = data[:, :n//4]
        data2 = data[:, n//4:3*n//4+1]
        data3 = data[:, 3*n//4+1:]
        container.d[f"meta/{self.target}/MMAV1"] = (0.5 * np.sum(np.abs(data1), axis=1)
                                         + np.sum(np.abs(data2), axis=1)
                                         + 0.5 * np.sum(np.abs(data3), axis=1)) / n

        qsum = np.sum(data * data, axis=1)
        container.d[f"meta/{self.target}/SSI"] = qsum
        container.d[f"meta/{self.target}/VAR"] = qsum / (n-1)
        container.d[f"meta/{self.target}/RMS"] = np.sqrt(qsum / n)

        df = np.abs(data[:, 1:] - data[:, :-1])
        container.d[f"meta/{self.target}/WL"] = np.sum(df, axis=1)
        container.d.make_folder(f"meta/{self.target}/WAMP")
        container.d[f"meta/{self.target}/WAMP"].update(
            {str(t): np.sum(np.where(df >= t, 1, 0), axis=1) for t in self.wamp_threshold})

        container.d[f"meta/{self.target}/LOG"] = np.exp(np.sum(np.log(np.abs(data)), axis=1) / n)

        return container

    def log(self) -> str:
        return f"fex({self.target})"


class MarkerSplitter(SigOperator):
    def __init__(self, annotation_spec: str, left_outer_segments: bool = False,
                 right_outer_segment: bool = False):
        self.aspec = annotation_spec
        self.left_segment = left_outer_segments
        self.right_segment = right_outer_segment

    def apply(self, container: SigContainer) -> Iterable[SigContainer]:
        limits = container.get_annotation_positions(self.aspec, TimeUnit.SAMPLE,
                                                    container.d["signals/fs"])
        if self.left_segment and limits[0] != 0:
            limits = np.insert(limits, 0, 0)
        if self.right_segment and limits[-1] != container.sample_count - 1:
            limits = np.append(limits, [container.sample_count - 1])
        return [self.container_factory(container, a, b)
                for a, b in zip(limits, limits[1:])]

    def container_factory(self, container: SigContainer, a: int, b: int):
        c = SigContainer.from_container(container, a, b)
        c.d["log"].append(f"msplit({self.aspec})@{a}:{b}")
        return c


class SimpleBranching(SigOperator):
    def __init__(self, branch: SigOperator):
        self.branch = branch

    @staticmethod
    def container_factory(container: SigContainer):
        c = SigContainer.from_container(container, 0, container.sample_count)
        return c


class Tee(SimpleBranching):
    def __init__(self, branch: SigOperator):
        super().__init__(branch)

    def apply(self, container: SigContainer) -> SigContainer:
        copy = SimpleBranching.container_factory(container)
        copy | self.branch
        return container


class AltOptional(SimpleBranching):
    def __init__(self, alternative: SigOperator):
        super().__init__(alternative)

    def  apply(self, container: SigContainer) -> Iterable[SigContainer]:
        copy = SimpleBranching.container_factory(container)
        acontainer = copy | self.branch
        return container, acontainer


class UfuncOnSignals(SigOperator):
    def __init__(self, ufunc):
        self.ufunc = ufunc

    def apply(self, container: SigContainer) -> SigContainer:
        container.d["signals/data"] = self.ufunc(container.d["signals/data"])
        return container

    def log(self):
        return f"ufunc%{self.ufunc.__name__}"


class Convolution(SigOperator):
    def __init__(self, v: np.ndarray):
        self.v = v
        self.sum = np.sum(self.v)

    def apply(self, container: SigContainer) -> SigContainer:
        result = np.empty_like(container.d["signals/data"])
        for i in range(container.channel_count):
            result[i] = np.convolve(container.d["signals/data"][i, :], self.v,
                                    mode="same") / self.sum
        container.d["signals/data"] = result
        return container

    def log(self):
        return f"conv({smart_tostring(self.v).strip()})"


class CrossCorrelation(SigOperator):
    def __init__(self, v: np.ndarray):
        self.v = v
        self.sum = np.sum(self.v)

    def apply(self, container: SigContainer) -> Union[SigContainer, Iterable[SigContainer]]:
        container.d["signals/data"] = sig.correlate(container.d["signals/data"],
                                                    self.v) / self.sum
        return container

    def log(self):
        return f"corr({smart_tostring(self.v).strip()})"


class Fft(SigOperator):
    def __init__(self, n: Optional[int] = None, target: str = "fft"):
        self.n = n
        self.target = "meta/" + target

    def apply(self, container: SigContainer) -> SigContainer:
        container.d[self.target + "/data"] = fft.rfft(container.d["signals/data"], self.n, axis=1)
        container.d[self.target + "/channels"] = container.d["signals/channels"]
        container.d[self.target + "/fs"] = container.d["signals/fs"]
        return container


class Hdf5(SigOperator):
    def __init__(self, filename):
        self.filename = filename

    @staticmethod
    def h5mapper(value):
        if isinstance(value, np.ndarray):
            if len(value) > 0 and isinstance(value[0], str):
                return "str_ndarray", np.array(value, dtype="S")
            return "ndarray", value
        if isinstance(value, list):
            if len(value) > -0 and isinstance(value[0], str):
                return "str_list", np.array(value, dtype="S")
            return "list", np.array(value)
        if isinstance(value, float):
            return "float", np.full((1,), value, dtype=np.float)
        if isinstance(value, int):
            return "int", np.full((1,), value, dtype=np.int)

    def apply(self, container: SigContainer) -> SigContainer:
        import h5py
        file = self.filename.format(container)
        with h5py.File(file, "w") as f:
            for path, value in container.d:
                type, hvalue = Hdf5.h5mapper(value)
                f[path] = hvalue
                f[path].attrs["type"] = type
        return container

    @staticmethod
    def load(filename):
        SigContainer.from_hdf5(filename)


class PResample(SigOperator):
    """
    Resampling using polyphase filtering
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
    """
    def __init__(self, *, up: int = None, down: int = None,
                 new_freq: Optional[Union[float, int]] = None):
        """
        Resampling of signal to sample frequency up * actual_frequency / down (exactly) or to
        new_freq (approximation with small up and down scales is used)
        :param up: upscaling parameter (only int are supported)
        :param down: downscaling parameter (inly int is supported)
        :param new_freq: target new frequency (optimal approximate fraction up/down is used)
        """
        if up is not None and down is not None and new_freq is None :
            self.up = up
            self.down = down
            self.new_freq = None
        elif up is None and down is None and new_freq is not None:
            self.up = None
            self.down = None
            self.new_freq = new_freq
        else:
            raise AttributeError("Invalid parameters")

    def apply(self, container: SigContainer) -> SigContainer:
        if self.new_freq is not None:
            f = fractions.Fraction(self.new_freq / container.d["signals/fs"]).limit_denominator(100)
            self.up = f.numerator
            self.down = f.denominator
        container.d["signals/data"] = sig.resample_poly(container.d["signals/data"],
                                                        self.up, self.down, axis=1)
        container.d["signals/fs"] = self.up * container.d["signals/fs"] / self.down
        andict = container.d["annotations"]
        for ann in andict:
            andict[ann]["samples"] = [self.up * sample // self.down
                                      for sample in andict[ann]["samples"]]
        return container

    def log(self):
        return (f"resample({self.up}/{self.down})" if self.up is not None
                else f"resample({self.new_freq})")
