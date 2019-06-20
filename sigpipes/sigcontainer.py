import re
from enum import Enum
from typing import Sequence, Any, Union
from re import split

import numpy as np
from sigpipes.auxtools import type_info, smart_tostring
import h5py


class TimeUnit(Enum):
    SAMPLE = 0
    SECOND = 1
    TIME_DELTA = 2

    def fix(self):
        return TimeUnit.SECOND if self == TimeUnit.TIME_DELTA else self


class HierarchicalDict:
    """
    Dictionary with hierarchical keys implemented by structure od nested (standard) dictionaries.
    Individual levels in hierarchical keys are separated by slashes (initial or final slashes
    are optional).
    """
    def __init__(self):
        self.root = {}

    def _create_path(self, key: str):
        path = split(r"/+", key.strip("/"))
        adict = self.root
        for folder in path[:-1]:
            if folder not in adict:
                adict[folder] = {}
            if not isinstance(adict[folder], dict):
                raise KeyError(f"{folder} in {key} is leaf not folder")
            adict = adict[folder]
        return path, adict

    def __setitem__(self, key: str, value: Any) -> None:
        """
        Set value with given hierarchical key (path). It creates all levels in the
        path. The value can have any type except "dict" (empty folders can be created
        by `make_folder` method)
        """
        assert not isinstance(value, dict), "dict is invalid leaf value"
        path, adict = self._create_path(key)
        leaf = path[-1]
        if isinstance(adict.get(leaf, None), dict):
            raise KeyError(f"leaf name {leaf} in {key} is folder")
        adict[leaf] = value

    def __getitem__(self, key: str) -> Any:
        """
        Get value with given hierarchical key (path).
        Empty path is invalid.
        """
        path = split(r"/+", key.strip("/"))
        adict = self.root
        for folder in path[:-1]:
            if folder not in adict:
                raise KeyError(f"folder {folder} in key {key} does not exist")
            adict = adict[folder]
        leaf = path[-1]
        if leaf not in adict:
            raise KeyError(f"leaf {leaf} in key {key} does not exist")
        return adict[leaf]

    def __contains__(self, key: str) -> bool:
        """
        Testing existence of given hierarchical key (in the form of leaf value or
        folder).
        """
        path = split(r"/+", key.strip("/"))
        adict = self.root
        for folder in path:
            if folder not in adict:
                return False
            adict = adict[folder]
        return True

    def make_folder(self, key: str) -> None:
        """
        Creation of empty folder with given path (all levels of path are
        created)
        """
        path, adict = self._create_path(key)
        folder_name = path[-1]
        if folder_name in adict:
            raise KeyError(f"key {folder_name} exists")
        adict[folder_name] = {}

    def __str__(self):
        return HierarchicalDict.rec_print(self.root, 0)

    def __repr__(self):
        return HierarchicalDict.rec_print(self.root, 0)

    @staticmethod
    def rec_print(d, level: int):
        return "\n".join(
            f"{'  ' * level}{key}: {type_info(value)} {smart_tostring(value, level + len(key))}"
            if not isinstance(value, dict)
            else f"{'  ' * level}{key}/\n{HierarchicalDict.rec_print(value, level + 1)}"
            for key, value in d.items())

    def __iter__(self):
        yield from self.rec_iterator(self.root, "")

    def rec_iterator(self, d, prefix):
        for key, value in d.items():
            path = prefix + "/" + key
            if isinstance(value, dict):
                yield from self.rec_iterator(d[key], path)
            else:
                yield path, value


class SigContainer:
    """
    Hierarchical container for physiological signal data (including annotations and
    auxiliary (meta)data).
    """
    def __init__(self, data: HierarchicalDict) -> None:
        """
        Private constructor. Use factory methods: from_signal_array or from_hdf5.
        """
        self.d = data

    @staticmethod
    def from_container(container: "SigContainer", a: int, b: int):
        d = HierarchicalDict()
        d["signals/data"] = container.d["signals/data"][:, a:b]
        d["signals/channels"] = container.d["signals/channels"]
        d["signals/units"] = container.d["signals/units"]
        d["signals/fs"] = container.d["signals/fs"]

        d["log"] = list(container.d["log"])
        if "annotations" in container.d:
            anns = SigContainer.cut_annots(container.d["annotations"], a, b)
            d.make_folder("annotations")
            d["annotations"].update(anns)
        d.make_folder("meta")
        d["meta"].update(container.d["meta"])
        return SigContainer(d)

    @staticmethod
    def from_signal_array(signals: np.ndarray, channels: Sequence[str],
                  units: Sequence[str], fs: float = 1.0) -> "SigContainer":
        """
        :param signals: signals as 2D array, channels in rows
        :param channels: identifiers of channels
        :param units:  units of channels data
        :param fs: (common) sampling frequency
        """
        d = HierarchicalDict()
        d["signals/data"] = signals
        d["signals/channels"] = channels
        d["signals/units"] = units
        d["signals/fs"] = fs

        d["log"] = []
        d.make_folder("meta")
        return SigContainer(d)

    def add_annotation(self, annotator: str,
                       samples: Sequence[int], types: Sequence[str],
                       notes: Sequence[str]) -> "SigContainer":
        """
        Add set of annotations to container.
        See https://physionet.org/physiobank/annotations.shtml for detailed description.
        :param annotator: short identifier of source of this annotations
        :param samples:  list of annotated samples in signal
        :param types:  list of short (one character) identification of individual annotations
                       (one per list of samples)
        :param notes:  longer description of individual annotations
                       (one per record in list of samples)
        """
        self.d[f"annotations/{annotator}/samples"] = samples
        self.d[f"annotations/{annotator}/types"] = types
        self.d[f"annotations/{annotator}/notes"] = notes
        return self

    def __getitem__(self, key: str) -> Any:
        """
        getter for container data (signals, annotations, auxiliary data)
        """
        return self.d[key]

    @property
    def signals(self):
        return self.d["signals/data"]

    def __str__(self):
        return str(self.d)

    def get_channel_triple(self, i):
        """
        get information for i-th channel
        :return: triple (data of channel, channel id, channel unit)
        """
        return (self.d["signals/data"][i, :], self.d["signals/channels"][i],
                self.d["signals/units"][i])

    @property
    def sample_count(self):
        """
        :return: number of samples in signal
        """
        return self.d["signals/data"].shape[1]

    @property
    def channel_count(self):
        """
        :return: numberf of channels
        """
        return self.d["signals/data"].shape[0]

    @property
    def id(self):
        """
        :return: unique identifier of container state (sequence of modification performed
        on container). This identifier can be used as unique filename for outputs.
        """
        return "~".join(op for op in self.d["log"]
                        if not op.startswith("#")).replace(".", ",").replace(" ", "")

    def x_index(self, index_type: TimeUnit, fs: Union[int, float]):
        """
        X-axis array for signal data in given time units (time representation).
        """
        if index_type == TimeUnit.SAMPLE:
            index = np.arange(0, self.sample_count)
        else:
            interval = 1.0 / fs
            index = np.linspace(0.0, self.sample_count * interval, self.sample_count,
                                endpoint=False)
            if index_type == TimeUnit.TIME_DELTA:
                index = np.fromiter((np.timedelta64(int(t * 1_000_000_000), "ns") for t in index),
                                    dtype="timedelta64[ns]")
        return index

    def get_annotation_positions(self, specifier: str, index_type: TimeUnit,
                                 fs: Union[int, float]):
        """
        Positions of annotations (markers) in given time units (time representation)
        :param specifier: identification of annotation source (annotator)
        :param index_type: time line representation
        :param fs: sample frequency
        :return:
        """
        annotator, achar, astring = SigContainer.annotation_parser(specifier)
        if annotator not in self.d["annotations"]:
            raise KeyError(f"""Invalid annotator {annotator}
                           (only {','.join(self.d['annotations'].keys())} are included)""")
        samples = np.array(self.d[f"annotations/{annotator}/samples"])
        types = np.array(self.d[f"annotations/{annotator}/types"])
        notes = np.array(self.d[f"annotations/{annotator}/notes"])
        if achar is not None:
            samples = samples[types == achar]
            notes = notes[types == achar]
        if astring is not None:
            r = re.compile(astring)
            vmatch = np.vectorize(lambda text: bool(r.fullmatch(text)))
            samples = samples[vmatch(notes)]
        return self.x_index(index_type, fs)[samples]

    @staticmethod
    def annotation_parser(aname: str) -> Sequence[str]:
        match = re.fullmatch(r"(\w+)(?:/(.))?(?:=(.+))?", aname)
        if match:
            return match.groups()
        else:
            raise KeyError(f"Invalid annotation specification `{aname}`")

    def get_fft_tuple(self, i: int, source: str):
        return self.d[f"{source}/data"][i, :], self.d[f"{source}/channels"][i]

    @staticmethod
    def from_hdf5(filename: str) -> "SigContainer":
        data = HierarchicalDict()
        with h5py.File(filename, "r") as f:
            f.visititems(lambda name, item : SigContainer._visitor(data, name, item))
        return SigContainer(data)

    @staticmethod
    def _visitor(data: HierarchicalDict, name: str, item: Union[h5py.Group, h5py.Dataset]) -> None:
        if isinstance(item, h5py.Group):
            data.make_folder(name)
        elif isinstance(item, h5py.Dataset):
            type = item.attrs["type"]
            if type == "int":
                data[name] = int(item[0])
            elif type == "float":
                data[name] = float(item[0])
            elif type == "str":
                data[name] = str(item[0])
            elif type in ["ndarray", "list", "str_ndarray", "str_list"]:
                data[name] = item.value

    @staticmethod
    def cut_annots(adict, start_sample, end_sample):
        tdict = {}
        for annotator in adict.keys():
            samples = np.array(adict[annotator]["samples"], copy=True)
            mn, mx = np.searchsorted(samples, [start_sample, end_sample])
            tdict[annotator] = {}
            tdict[annotator]["samples"] = samples[mn:mx] - start_sample
            tdict[annotator]["types"] = adict[annotator]["types"][mn:mx]
            tdict[annotator]["notes"] = adict[annotator]["notes"][mn:mx]
        return tdict
