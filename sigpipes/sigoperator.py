from sigpipes import features
from sigpipes.sigcontainer import SigContainer, DPath
from sigpipes.auxtools import seq_wrap
from sigpipes.auxtools import TimeUnit
import gzip

from typing import Sequence, Union, Iterable, Optional, MutableMapping, Any, Mapping
import collections.abc
import sys
import fractions
from pathlib import Path

import numpy as np
import scipy.signal as sig
import scipy.fftpack as fft


class SigOperator:
    """
    Base abstract class of signal operators.
    """
    def apply(self, container: SigContainer) -> Any:
        raise NotImplementedError("Abstract method")

    def prepare_container(self, container: SigContainer) -> SigContainer:
        """
        Prepare container at the beginning of apply method.
        (this method must be called at the first line of `apply` method)

        Args:
            container: prepared signal container
        """
        return container

    def __ror__(self, container: Union[SigContainer, Sequence[SigContainer], "SigOperator"]
                ) -> Any:
        """
        Pipe operator for streamlining of signal operators

        Args:
            container:  left operand i.e signal container (input), sequence of containers
            (multiple inputs) or another signal operator (formation of compound operators).

        Returns:
            - for container as input: container, sequence of containers,
              or another data structures (only consumers)
            - for sequence of containers as input: sequence of containers,
              sequence of another data structures (only consumers)
            - for signal operators in both operands:  compound signal operator
        """
        if isinstance(container, SigContainer):
            container.d["log"].append(self.log())
            return self.apply(container)
        elif isinstance(container, collections.abc.Sequence):
            return [c | self for c in container]
        elif isinstance(container, SigOperator):
            return CompoundSigOperator(container, self)
        else:
            raise TypeError("Unsupported left operand of pipe")

    def __or__(self, other):
        return CompoundSigOperator(self, other)

    def log(self):
        """
        Identification of operation for logging purposes.

        Returns:
        Simple (and if possible short) identification.
        """
        return self.__class__.__name__


class Identity(SigOperator):
    """
    Base class for operators which do not modify container.
    """
    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        return container

    def log(self):
        return "#" + self.__class__.__name__


class MaybeConsumerOperator(Identity):
    """
    Abstract class for operators which can works as final consumers i.e. it can produce different
    representation of signal data e.g. dataframes, matplot figures, etc.
    """
    pass


class CompoundSigOperator(SigOperator):
    def __init__(self, left_operator: SigOperator, right_operator: SigOperator) -> None:
        self.left = left_operator
        self.right = right_operator

    def apply(self, container: SigContainer):
        container = self.prepare_container(container)
        return container | self.left | self.right

    def log(self):
        return "#COMP"


class Print(Identity):
    """
    Operator which prints debug text representation into text output
    """
    def __init__(self, output=sys.stdout, header=True):
        """
        Args:
            output: file-like object in text mode
            header: the header with log-id is printed
        """
        if isinstance(output, str):
            self.output = open(output, "wt")
        else:
            self.output = output
        self.header = header

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        if self.header:
            print(container.id, file=self.output)
            print("-"*40, file=self.output)
        print(str(container), file=self.output)
        return container


class SigModifierOperator(SigOperator):
    """
    Abstract class for operators which modify signal data.
    """
    def prepare_container(self, container: SigContainer) -> SigContainer:
        return SigContainer(container.d.deepcopy(shared_folders=["annotations"],
                                                 empty_folders=["meta"]))


class Sample(SigModifierOperator):
    """
    Sample (continuous interval) of signal (for all channels)
    """
    def __init__(self, start: Union[int, float, np.timedelta64],
                 end: Union[int, float, np.timedelta64]):
        """
        Args:
            start: start point of sample. integer: sample number, float: time in seconds,
                   np.timedelta64: time represented by standard time representation of numpy)
            end: end point of sample (see `start` for interpretation)
        """
        self.start = start
        self.end = end

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        fs = container.d["signals/fs"]
        lag = container.lag
        start = TimeUnit.to_sample(self.start, fs, TimeUnit.time_unit_mapper(self.start), lag)
        end = TimeUnit.to_sample(self.end, fs, TimeUnit.time_unit_mapper(self.end), lag)
        container.d["signals/data"] = container.d["signals/data"][:, start:end]
        container.d["signals/lag"] = lag - start

        if "annotations" in container.d:
            adict = container.d["annotations"]
            newdict = SigContainer.cut_annots(adict, start, end)
            adict.update(newdict)

        return container

    def log(self):
        return f"SAMP@{str(self.start)}@{str(self.end)}"


class ChannelSelect(SigOperator):
    """
    Selection of limited subset of channels.
    """
    def __init__(self, selector: Sequence[int]) -> None:
        """
        Args:
            selector: sequence of (integer) indexes of channels
        """
        self.selector = selector

    def prepare_container(self, container: SigContainer) -> SigContainer:
        return SigContainer(container.d.deepcopy(shared_folders=["annotations"],
                                                 empty_folders=["signals"]))

    def apply(self, container: SigContainer) -> SigContainer:
        nc = self.prepare_container(container)
        nc.d["signals/data"] = container.d["signals/data"][self.selector, :]
        nc.d["signals/channels"] = np.array(container.d["signals/channels"])[self.selector].tolist()
        nc.d["signals/units"] = np.array(container.d["signals/units"])[self.selector].tolist()
        nc.d["signals/fs"] = container.d["signals/fs"]
        if "meta" in nc.d:
            nc.d.map(lambda a: a[self.selector], root="meta")
        return nc

    def log(self):
        return f"CHSEL@{','.join(str(s) for s in self.selector)}"


class MetaProducerOperator(SigOperator):
    """
    Abstract class for operators which product metadata (i.e. data inferred from signals)
    """
    def prepare_container(self, container: SigContainer) -> SigContainer:
        return SigContainer(container.d.deepcopy(["signals", "annotation"]))


class FeatureExtractor(MetaProducerOperator):
    def __init__(self, features_dict: Mapping[str,Union[bool,float,Sequence[float]]] = None,
                 *, wamp_threshold: Union[float, Sequence[float]] = (),
                 zc_diff_threshold: float = (), zc_mul_threshold = (),
                 sc_threshold: float = ()):
        self.feature_dict = features_dict if features_dict is not None else {feature: True for feature
                                                                                  in features.NON_THRESHOLD}
        if wamp_threshold:
            self.feature_dict["WAMP"] = wamp_threshold
        if zc_diff_threshold and zc_mul_threshold:
            self.feature_dict["ZC"] = zip(zc_diff_threshold, zc_mul_threshold)
        if sc_threshold:
            self.feature_dict["SC"] = sc_threshold

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        n = container.sample_count
        data = container.d["signals/data"]
        fkeys = {key for key in self.feature_dict.keys() if self.feature_dict[key]}
        thresholds = {key : value for key,value in self.feature_dict.items() if key in features.WITH_THRESHOLD}
        fdict = features.features(data, fkeys, thresholds)
        path = "meta/features"
        container.d.make_folder(path)
        container.d[path].update(fdict)
        return container


class FeatureExtraction(MetaProducerOperator):
    """
    Extraction of basic features of signal.
    """
    def __init__(self, *, wamp_threshold: Union[float, Sequence[float]] = (),
                 zc_diff_threshold: float = 0.0, zc_mul_threshold = 0.0,
                 sc_threshold: float = 0.0):
        """
        Args:
            wamp_threshold:  threshold value (or sequence of values) foe WAMP feature
        """
        self.wamp_threshold = seq_wrap(wamp_threshold)
        self.target = "features"
        self.zc_diff_threshold = zc_diff_threshold
        self.zc_mul_threshold = zc_mul_threshold
        self.sc_threshold = sc_threshold

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        n = container.sample_count
        data = container.d["signals/data"]
        absum = np.sum(np.abs(data), axis=1)
        container.d[f"meta/{self.target}/IEMG"] = absum
        container.d[f"meta/{self.target}/MAV"] = absum / n

        data1 = np.abs(data[:, :n//4])
        data2 = np.abs(data[:, n//4:3*n//4+1])
        data3 = np.abs(data[:, 3*n//4+1:])
        wsum = np.sum(data2, axis=1)
        container.d[f"meta/{self.target}/MMAV1"] = (
            (0.5 * np.sum(data1, axis=1) + wsum + 0.5 * np.sum(data3, axis=1)) / n)

        koef1 = 4 * np.arange(1, n//4 + 1, dtype=np.float64) / n
        koef3 = 4 * (np.arange(3*n//4 + 2, n+1, dtype=np.float64) - n) / n

        container.d[f"meta/{self.target}/MMAV2"] = (
            (np.sum(koef1 * data1, axis=1) + wsum + np.sum(koef3 * data3, axis=1)) / n)

        qsum = np.sum(data * data, axis=1)
        container.d[f"meta/{self.target}/SSI"] = qsum
        container.d[f"meta/{self.target}/VAR"] = qsum / (n-1)
        container.d[f"meta/{self.target}/RMS"] = np.sqrt(qsum / n)

        df = np.abs(data[:, :-1] - data[:, 1:])
        container.d[f"meta/{self.target}/WL"] = np.sum(df, axis=1)
        container.d.make_folder(f"meta/{self.target}/WAMP")
        container.d[f"meta/{self.target}/WAMP"].update(
            {str(t): np.sum(np.where(df >= t, 1, 0), axis=1) for t in self.wamp_threshold})

        container.d[f"meta/{self.target}/LOG"] = np.exp(np.sum(np.log(np.abs(data)), axis=1) / n)

        container.d[f"meta/{self.target}/ZC"] = np.sum(
            np.where(np.logical_and(data[:, :-1] * data[:, 1:] >= self.zc_mul_threshold,
                                    df >= self.zc_diff_threshold), 1, 0), axis=1)

        container.d[f"meta/{self.target}/SC"] = np.sum(
            np.where((data[:, 1:-1] - data[:, :-2]) * (data[:, 1:-1] - data[:, 2:])
                     >= self.sc_threshold, 1, 0), axis=1)

        return container

    def log(self) -> str:
        return f"FEX"


class SplitterOperator(SigOperator):
    """
    Abstract class for splitters (i.e. operators which split container into several containers
    (segments) that can be processes independently as sequence of container.
    """
    def container_factory(self, container: SigContainer, a: int, b: int,
                          splitter_id: str) -> SigContainer:
        c = SigContainer(container.d.deepcopy(empty_folders=["meta", "annotations"]))
        c.d["signals/data"] = c.d["signals/data"][:, a:b]
        newlog = list(c.d["log"])
        newlog.append(f"{splitter_id}@{a}-{b}")
        c.d["log"] = newlog
        if "annotations" in container.d:
            c.d["annotations"].update(SigContainer.cut_annots(container.d["annotations"], a, b))
        return c


class SampleSplitter(SplitterOperator):
    """
    Splitting of signals data to several containers in points defined by samples
    or its absolute time. Only inner intervals are included!
    The returned data can be processes independently as sequence of container.
    """
    def __init__(self, points: Sequence[Union[int, float, np.timedelta64]]) -> None:
        self.points = points

    def apply(self, container: SigContainer) -> Sequence[SigContainer]:
        container = self.prepare_container(container)
        fs = container.d["signals/fs"]
        limits = [TimeUnit.to_sample(point, fs, TimeUnit.time_unit_mapper(point))
                  for point in self.points]
        limits.sort()
        return [self.container_factory(container, a, b, "SPL")
                for a, b in zip(limits, limits[1:])]


class MarkerSplitter(SplitterOperator):
    """
    Splitting of signals data to several containers in points defined by annotation (marker).
    The returned data can be processes independently as sequence of container.
    """
    def __init__(self, annotation_spec: str, left_outer_segments: bool = False,
                 right_outer_segment: bool = False) -> None:
        """
        Args:
            annotation_spec: specification of splitting annotations (annotator)
            left_outer_segments: true = signal before the first splitting annotation is included
            right_outer_segment: true = signal after the last splitting annotation is included
        """
        self.aspec = annotation_spec
        self.left_segment = left_outer_segments
        self.right_segment = right_outer_segment

    def apply(self, container: SigContainer) -> Sequence[SigContainer]:
        container = self.prepare_container(container)
        limits = container.get_annotation_positions(self.aspec, TimeUnit.SAMPLE,
                                                    container.d["signals/fs"])
        if self.left_segment and limits[0] != 0:
            limits = np.insert(limits, 0, 0)
        if self.right_segment and limits[-1] != container.sample_count - 1:
            limits = np.append(limits, [container.sample_count])
        return [self.container_factory(container, a, b, f"MSPL@{self.aspec}]")
                for a, b in zip(limits, limits[1:])]


class SimpleBranching(SigOperator):
    """
    Abstract class for branching operators i.e  operators bifurcating stream to two or more branches
    which are initially identical (based on the same container).
    """
    def __init__(self,   *branches):
        self.branches = branches

    @staticmethod
    def container_factory(container: SigContainer):
        nc = SigContainer(container.d.deepcopy())
        nc.d["log"] = list(nc.d["log"])
        return nc


class Tee(SimpleBranching):
    """
    Tee branching operator. For each parameters of constructor the container is duplicated
    and processed by pipeline passed by this parameter (i.e. all pipelines have the same source,
    but they are independent). Only original container is returned (i.e. only one stream continues)
    """
    def __init__(self, *branches):
        """
        Args:
            *branches:  one or more parameters in the form of signals operators (including whole
            pipelines in the form of compound operator)
        """
        super().__init__(*branches)

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        for branch in self.branches:
            copy = SimpleBranching.container_factory(container)
            copy | branch
        return container

    def log(self):
        return "#TEE"


class VariantsSplitter(SimpleBranching):
    pass


class AltOptional(VariantsSplitter):
    """
    Alternative branching operator.  For each parameters of constructor the container is duplicated
    and processed by pipeline passed by this parameter (i.e. all pipelines have the same source,
    but they are independent).
    List of containers are returned including original containers and all processed
    duplicates.
    """
    def __init__(self, *alternatives):
        """
        Args:
            *branches:  one or more parameters in the form of signals operators (including whole
            pipelines in the form of compound operator)
        """
        super().__init__(*alternatives)

    def apply(self, container: SigContainer) -> Sequence[SigContainer]:
        container = self.prepare_container(container)
        acontainer = [container]
        for branch in self.branches:
            copy = SimpleBranching.container_factory(container)
            acontainer.append(copy | branch)
        return acontainer

    def log(self):
        return "#ALTOPT"


class Alternatives(VariantsSplitter):
    """
    Alternative branching operator.  For each parameters of constructor the container is duplicated
    and processed by pipeline passed by this parameter (i.e. all pipelines have the same source, but they are
    independent). List of containers are returned including all processed duplicates.
    """

    def __init__(self, *alternatives):
        super().__init__(*alternatives)

    def apply(self, container: SigContainer) -> Sequence[SigContainer]:
        container = self.prepare_container(container)
        acontainer = []
        for branch in self.branches:
            copy = SimpleBranching.container_factory(container)
            acontainer.append(copy | branch)
        return acontainer

    def log(self):
        return "#ALT"


class UfuncOnSignals(SigModifierOperator):
    """
    Application of unary numpy ufunc on signals.

    Examples:
        container | UfuncOnSignals(np.abs)
    """
    def __init__(self, ufunc):
        self.ufunc = ufunc

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        container.d["signals/data"] = self.ufunc(container.d["signals/data"])
        return container

    def log(self):
        if hasattr(self.ufunc, "__name__"):
            return f"UF@{self.ufunc.__name__}"
        else:
            return "UF"


class Scale(SigModifierOperator):
    """
    Scale signal by scalar.

    Examples:
        container | Scale(-1.0)
    """
    def __init__(self, scalar: float):
        self.scalar = scalar

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        container.d["signals/data"] = self.scalar * container.d["signals/data"]
        return container

    def log(self):
        return f"{self.scalar}x"


class MVNormalization(SigModifierOperator):
    """
        Mean and variance normalization
    """
    def __init__(self, mean: Optional[float] = 0.0, variance: Optional[float] = 1.0):
        self.mean = mean
        self.variance = variance

    def apply(self, container: SigContainer) -> SigContainer:
        if self.mean is not None:
            mean = np.mean(container.d["signals/data"], axis=1).reshape(container.channel_count, 1)
            if self.mean == 0:
                container.d["signals/data"] -= mean
            else:
                container.d["signals/data"] -= mean - self.mean
        if self.variance is not None:
            variance = np.var(container.d["signals/data"], axis=1).reshape(container.channel_count, 1)
            if self.variance == 1.0:
                container.d["signals/data"] /= variance
            else:
                container.d["signals/data"] /= variance / self.variance
        return container

    def log(self):
        return f"MVNorm@{self.mean},{self.variance}"


class RangeNormalization(SigModifierOperator):
    """
        Normalize signal to range <a,b>.
    """
    def __init__(self, min=0, max=1.0):
        assert min < max
        self.min = min
        self.max = max

    def apply(self, container: SigContainer) -> Any:
        dmax = np.max(container.signals, axis=1).reshape(container.channel_count, 1)
        dmin = np.min(container.signals, axis=1).reshape(container.channel_count, 1)
        drange = (dmax - dmin).reshape(container.channel_count, 1)
        range = self.max - self.min
        container.d["signals/data"] = self.min + range * (container.signals - dmin) / drange
        container.d["signals/units"] = ["unit"] * container.channel_count
        return container

    def log(self):
        return f"RangeNorm@{self.min},{self.max}"


class Convolution(SigModifierOperator):
    """
    Convolution of signal data (all signals)
    """
    def __init__(self, v: Sequence[float]):
        self.v = np.array(v, dtype=np.float)
        self.sum = np.sum(self.v)

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        result = np.empty_like(container.d["signals/data"])
        for i in range(container.channel_count):
            result[i] = np.convolve(container.d["signals/data"][i, :], self.v,
                                    mode="same") / self.sum
        container.d["signals/data"] = result
        return container

    def log(self):
        return f"CONV@{len(self.v)}"


class CrossCorrelation(SigModifierOperator):
    def __init__(self, v: np.ndarray):
        self.v = v
        self.sum = np.sum(self.v)

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        result = np.empty_like(container.d["signals/data"])
        for i in range(container.channel_count):
            result[i] = np.correlate(container.d["signals/data"][i, :], self.v,
                                     mode="same") / self.sum
        container.d["signals/data"] = result
        return container

    def log(self):
        return f"CORR@{len(self.v)}"


class Fft(MetaProducerOperator):
    def __init__(self, n: Optional[int] = None, target: str = "fft"):
        self.n = n
        self.target = "meta/" + target

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        container.d[self.target + "/data"] = np.abs(fft.fft(container.d["signals/data"], self.n, axis=1))
        container.d[self.target + "/channels"] = [name + " (spectre)" for name in container.d["signals/channels"]]
        return container

    def log(self):
        return "#FFT"


class FFtAsSignal(SigModifierOperator):
    def __init__(self, fftSection: str = "fft"):
        self.target = "meta/" + fftSection

    def apply(self, container: SigContainer) -> Any:
        data = container.d[self.target + "/data"]
        channels = container.d[self.target + "/channels"]
        container = self.prepare_container(container)
        container.d["signals/data"] = data
        container.d["signals/channels"] = channels
        return container


class Hdf5(Identity):
    """
    Serializer of containers to HDF5 file
    """
    def __init__(self, file:str = "", *, dir: str = ""):
        """
        Args:
            file: name of hdf5 file
        """
        self.filepath = DPath.from_path(file).prepend_path(DPath.from_path(dir, dir=True))

    @staticmethod
    def h5mapper(value):
        if isinstance(value, np.ndarray):
            if len(value) > 0 and isinstance(value[0], str):
                return "str_ndarray", np.array([s.encode(encoding="ascii", errors="backslashreplace") for s in value],
                                               dtype="S")
            return "ndarray", value
        if isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], str):
                return "str_list", np.array([s.encode(encoding="utf-8", errors="backslashreplace") for s in value],
                                            dtype="S")
            return "list", np.array(value)
        if isinstance(value, float):
            return "float", np.full((1,), value, dtype=np.float)
        if isinstance(value, int):
            return "int", np.full((1,), value, dtype=np.int)
        if isinstance(value, str):
            return "str", np.array([value.encode(encoding="utf-8", errors="backslashreplace")], dtype="S")
        else:
            raise TypeError(f"unsupported type {value.__class__} of value `{value}`")

    def apply(self, container: SigContainer) -> SigContainer:
        import h5py
        container = self.prepare_container(container)
        path = self.filepath.base_path(container.basepath.extend_stem(container.id).resuffix(".hdf5"))
        with h5py.File(str(path), "w") as f:
            for path, value in container.d:
                dtype, hvalue = Hdf5.h5mapper(value)
                f[path] = hvalue
                f[path].attrs["type"] = dtype
        return container

    def sigcontainer(self):
        SigContainer.from_hdf5(str(self.basepath))


class PResample(SigModifierOperator):
    """
    Resampling using polyphase filtering
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html)
    """
    def __init__(self, down: int = None, *, up: int = 1,
                 new_freq: Optional[Union[float, int]] = None):
        """
        Resampling of signal to sample frequency up * actual_frequency / down (exactly) or to
        new_freq (approximation with small up and down scales is used)

        Args:
            up: upscaling parameter (only int are supported)
            down: downscaling parameter (only int is supported)
            new_freq: target new frequency (optimal approximate fraction up/down is used)
        """
        if down is not None and new_freq is None :
            self.up = up
            self.down = down
            self.new_freq = None
        elif down is None and new_freq is not None:
            self.up = None
            self.down = None
            self.new_freq = new_freq
        else:
            raise AttributeError("Invalid parameters")

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        if self.new_freq is not None:
            f = fractions.Fraction(self.new_freq / container.d["signals/fs"]).limit_denominator(100)
            self.up = f.numerator
            self.down = f.denominator
        container.d["signals/data"] = sig.resample_poly(container.d["signals/data"],
                                                        self.up, self.down, axis=1)
        container.d["signals/fs"] = self.up * container.d["signals/fs"] / self.down
        if "annotations" in container.d:
            andict = container.d["annotations"]
            for ann in andict.keys():
                andict[ann]["samples"] = [self.up * sample // self.down
                                          for sample in andict[ann]["samples"]]
        return container

    def log(self):
        return (f"RSAM@{self.up}-{self.down}" if self.up is not None
                else f"RSAM@{self.new_freq}")


class Reaper(Identity):
    """
    Storage of containers or their fragments into dictionary
    """
    def __init__(self, store: MutableMapping[str, Any], store_key: str,
                 data_key: Optional[str] = None):
        """
        Args:
            store:  dictionary serving as storage
            store_key: key of saved data in dictionary
            data_key: path to part of hierarchical dictionary or None
                      (whole container is stored)
        """
        self.store = store
        self.skey = store_key
        self.dkey = data_key

    def apply(self, container: SigContainer) -> SigContainer:
        container = self.prepare_container(container)
        skey = self.skey.format(container)
        self.store[skey] = container[self.dkey] if self.dkey is not None else container
        return container


class Csv(Identity):
    """
    Serializer of containers to CSV file
    """
    def __init__(self, file: str = "", *, dir: str = "", dialect="excel",
                 time_unit: TimeUnit = TimeUnit.SECOND, gzipped=False):
        """
        Args:
            file: name of CSV file
        """
        self.filepath = DPath.from_path(file).prepend_path(DPath.from_path(dir, dir=True))
        if gzipped:
            self.filepath.add_suffix(".gz")
        self.dialect = dialect
        self.time_unit = time_unit
        self.gzipped =  gzipped

    def apply(self, container: SigContainer) -> SigContainer:
        import csv
        container = self.prepare_container(container)
        path = self.filepath.base_path(container.basepath.extend_stem(container.id).resuffix(".csv"))
        x = container.x_index(self.time_unit, container.d["signals/fs"])
        opener = open if not self.gzipped else gzip.open
        with opener(str(path), "wt", newline='') as csvfile:
            writer = csv.writer(csvfile, dialect=self.dialect)
            writer.writerow(["time"] + container.d["signals/channels"])
            for i in range(container.signals.shape[1]):
                writer.writerow([f"{val:g}" for val in
                    np.hstack((x[i], container.signals[:, i]))])
        return container
