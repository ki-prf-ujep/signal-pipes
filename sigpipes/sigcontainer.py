import re
from typing import Sequence, Any, Union, Dict, Optional, Callable
from re import split

import numpy as np
from sigpipes.auxtools import type_info, smart_tostring, TimeUnit
import h5py


def folder_copy(newdict: Dict[str, Any], olddict: Dict[str, Any],
                segpath: str, shared_folders: Sequence[str],
                empty_folders: Sequence[str]) -> None:
    for key, value in olddict.items():
        itempath = f"{segpath}/{key}" if segpath != "" else key
        if isinstance(value, dict):
            if itempath in empty_folders:
                newdict[key] = {}
            elif itempath in shared_folders:
                newdict[key] = olddict[key]
            else:
                newdict[key] = {}
                folder_copy(newdict[key], olddict[key], itempath, shared_folders, empty_folders)
        else:
            newdict[key] = value


def hdict_map(d: Dict[str, Any], function):
    for key, value in d.items():
        if isinstance(value, dict):
            hdict_map(value, function)
        else:
            d[key] = function(value)


class HierarchicalDict:
    """
    Dictionary with hierarchical keys implemented by structure od nested (standard) dictionaries.
    Individual levels in hierarchical keys are separated by slashes (initial or final slashes
    are optional).
    """
    def __init__(self, root: Dict[str, Any] = None):
        self.root = {} if root is None else root

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
        Get value with given hierarchical key (path). Empty path is invalid.
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
        Testing existence of given hierarchical key (in the form of leaf value or folder).
        """
        path = split(r"/+", key.strip("/"))
        adict = self.root
        for folder in path:
            if folder not in adict:
                return False
            adict = adict[folder]
        return True

    def deepcopy(self, shared_folders: Sequence[str]=[],
                 empty_folders: Sequence[str] = [],
                 root: Optional[str] = None) -> "HierarchicalDict":
        """
        Deep copy of folders of hierarchical tree or subtree (leaf values are not
        duplicated)

        Args:
            shared_folders: folders which are not duplicated but they are shared
            empty_folders: folders, which contents are not copied
                (i.e. folders are empty in duplicate]
            root: path to copied subtree (if None all tree is copied]

        Returns:
            duplicate or partial duplicate
        """
        new_hdict = {}
        folder_copy(new_hdict, self.root if root is None else self[root], "",
                    shared_folders, empty_folders)
        return HierarchicalDict(new_hdict)

    def make_folder(self, key: str) -> None:
        """
        Creation of empty folder with given path (all levels of path are
        created)

        Args:
            key: path to new folder
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

    def map(self, function: Callable[[Any], Any], root: Optional[str] = None):
        """
        Map (unary) function on all values of given subtree. Map changes original subtree
        (no duplication is performed).

        Args:
            function: mapping function (it must be defined for all values in subtree)
            root: path to subtree (or None if the whole hierarchical is modified)
        """
        hdict_map(self.root if root is None else self[root], function)

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
        Private constructor. Use factory methods: `from_signal_array` or `from_hdf5.`
        """
        self.d = data

    @staticmethod
    def from_signal_array(signals: np.ndarray, channels: Sequence[str],
                  units: Sequence[str], fs: float = 1.0) -> "SigContainer":
        """
        Creation of container from signal data (in numpy array) and basic metadata.

        Args:
            signals: signals as 2D array, channels in rows
            channels: identifiers of channels
            units:  units of channels data
            fs: (common) sampling frequency
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
        Add the set of annotations to container.
        See https://physionet.org/physiobank/annotations.shtml for detailed description.

        Args:
            annotator: short identifier of source of this annotations
            samples:  list of annotated samples in signal
            types:  list of short (one character) identification of individual annotations
                    (one per list of samples)
            notes:  longer description of individual annotations
                    (one per record in list of samples)
        """
        self.d[f"annotations/{annotator}/samples"] = samples
        self.d[f"annotations/{annotator}/types"] = types
        self.d[f"annotations/{annotator}/notes"] = notes
        return self

    def __getitem__(self, key: str) -> Any:
        """
        Getter for container data (signals, annotations, metadata, etc-)
        """
        return self.d[key]

    @property
    def signals(self):
        """
        Signal data (2D numpy array)
        """
        return self.d["signals/data"]

    def feature(self, name: str):
        return self.d[f"/meta/features/{name}"]

    def __str__(self):
        return str(self.d)

    def get_channel_triple(self, i):
        """
        Auxiliary getter for signal of  i-th channel with metadata.

        Returns:
            triple (data of channel, channel id, channel unit)
        """
        return (self.d["signals/data"][i, :], self.d["signals/channels"][i],
                self.d["signals/units"][i])

    @property
    def sample_count(self):
        """
        Number of samples in signal.
        """
        return self.d["signals/data"].shape[1]

    @property
    def channel_count(self):
        """
        Number of channels,
        """
        return self.d["signals/data"].shape[0]

    @property
    def id(self):
        """
            Unique identifier of container state (sequence of modifications performed
            on container). This identifier can be used as unique filename for outputs.
        """
        return "~".join(op for op in self.d["log"]
                        if not op.startswith("#")).replace(".", ",").replace(" ", "")

    def x_index(self, index_type: TimeUnit, fs: Union[int, float]):
        """
        Returns:
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
        Positions of annotations (markers) in given time units (time representation).

        Args:
            specifier: identification of annotation source (annotator)
            index_type: time line representation
            fs: sample frequency
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
