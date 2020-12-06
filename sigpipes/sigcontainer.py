import re
from typing import Sequence, Any, Union, Dict, Optional, Callable
from re import split
from pathlib import Path

import numpy as np
from sigpipes.auxtools import type_info, smart_tostring, TimeUnit
import h5py
import csv


class DPath:
    def __init__(self, root, dir, stem, suffix):
        self.root = root
        self.dir = dir
        self.stem = stem
        self.suffix = suffix

    @staticmethod
    def from_path(path, dir=False):
        p = Path(path)
        parts = Path(path).parts
        if p.is_absolute():
            root = parts[0]
            sind = 1
        else:
            root = ""
            sind = 0
        if dir:
            if sind < len(parts):
                dir = str(Path(parts[sind]).joinpath(*parts[sind+1:]))
            else:
                dir = ""
            stem = ""
            suffix = ""
        else:
            if sind < len(parts)-1:
                dir = str(Path(Path(parts[sind]).joinpath(*parts[sind+1:-1])))
            else:
                dir = ""
            if parts:
                suffix = "".join(Path(parts[-1]).suffix)
                stem = str(Path(parts[-1]))[:-len(suffix)]
            else:
                suffix = ""
                stem = ""
        return DPath(root, dir, stem, suffix)

    def extend_stem(self, extension, *, sep="_"):
        if extension == "":
            return self
        assert self.stem != ""
        return DPath(self.root, self.dir, self.stem + sep + extension, self.suffix)

    def resuffix(self, newsuffix):
        assert  self.stem != ""
        return DPath(self.root, self.dir, self.stem, newsuffix)

    def restem(self, newstem):
        return DPath(self.root, self.dir, newstem, self.suffix)

    def __repr__(self):
        return f"root: {self.root}, dir: {self.dir},  stem: {self.stem}, suffix: {self.suffix}"

    def __str__(self):
        return str(Path(self.root).joinpath(self.dir, self.stem+self.suffix))

    @property
    def empty(self):
        return self.root == "" and self.dir == "" and self.stem == "" and self.suffix == ""

    def prepend_path(self, prep):
        if prep.empty:
            return self
        if self.root:
            raise ValueError(f"Absolute path is not prependable {repr(prep)} < {self}")
        assert prep.stem == "" and prep.suffix == ""
        return DPath(prep.root, str(Path(prep.dir)/Path(self.dir)), self.stem, self.suffix)

    def base_path(self, base):
        if self.dir == "" and self.root == "":
            root = base.root
            dir = base.dir
        else:
            root = self.root
            dir = self.dir
        if self.stem == "":
            stem = base.stem
            suffix = base.suffix
        else:
            stem = self.stem
            suffix = self.suffix
        return DPath(root, dir, stem, suffix)


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
                  units: Sequence[str], fs: float = 1.0, basepath: str = "") -> "SigContainer":
        """
        Creation of container from signal data (in numpy array) and basic metadata.

        Args:
            basepath: base path for filenames
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
        d["basepath"] = str(DPath.from_path(basepath))
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
    def basepath(self):
        return DPath.from_path(self.d["basepath"])

    @property
    def id(self):
        """
            Unique identifier of container state (sequence of modifications performed
            on container). This identifier can be used as unique filename for outputs.
        """
        return "~".join(op for op in self.d["log"]
                        if not op.startswith("#")).replace(".", ",").replace(" ", "")

    @property
    def lag(self):
        return self.d["/signals/lag"] if "/signals/lag" in self.d else 0

    def x_index(self, index_type: TimeUnit, fs: Union[int, float]):
        """
        Returns:
            X-axis array for signal data in given time units (time representation).
        """
        if index_type == TimeUnit.SAMPLE:
            index = np.arange(0, self.sample_count) - self.lag
        else:
            interval = 1.0 / fs
            index = np.linspace(0.0, self.sample_count * interval, self.sample_count,
                                endpoint=False) - self.lag * interval
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
        match = re.fullmatch(r"([.\w]+)(?:/(.))?(?:=(.+))?", aname)
        if match:
            return match.groups()
        else:
            raise KeyError(f"Invalid annotation specification `{aname}`")

    def get_fft_tuple(self, i: int, source: str):
        return self.d[f"{source}/data"][i, :], self.d[f"{source}/channels"][i]

    @staticmethod
    def from_hdf5(filename: str, *, path: str = None, use_saved_path: bool = False) -> "SigContainer":
        data = HierarchicalDict()
        with h5py.File(filename, "r") as f:
            f.visititems(lambda name, item : SigContainer._visitor(data, name, item))
        if not use_saved_path:
            data["basepath"] = str(DPath.from_path(filename).prepend_path(DPath.from_path(path, dir=True)))
        return SigContainer(data)

    @staticmethod
    def hdf5_cache(source, operator: "SigOperator", path: str = "") -> "SigContainer":
        path = DPath.from_path(path).base_path(source.filepath.extend_stem("_cache").resuffix(".hdf5"))
        if Path(str(path)).exists():
            return SigContainer.from_hdf5(str(path), use_saved_path=True)
        else:
            from  sigpipes.sigoperator import Hdf5
            return source.sigcontainer() | operator| Hdf5(str(path))

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
                data[name] = item[0].decode(encoding="utf-8")
            elif type == "list":
                data[name] = list(item.value)
            elif type == "str_list":
                data[name] = list(s.decode(encoding='UTF-8') for s in item.value)
            elif type in ["ndarray", "str_ndarray"]:
                data[name] = item.value

    @staticmethod
    def from_csv(filename: str, * ,  dir: str = None, dialect: str = "excel", header: bool = True,
                 default_unit: str = "unit", fs=None, transpose: bool = False,
                 annotation: Union[str,Sequence[str]] = None) -> "SigContainer":
        """

        Args:
            filename: absolute or relative file name
            dir: base directory of relative file name (optional)
            dialect: dialect of CSV (see CSV module)
            header: first line is header with channels names and units (in parenthesis)
            default_unit: default unit (if is not defined by header)
            fs: sampling frequency, if smapling frequency is not provided, the frequency is
                derived from first column which must containts second timestamps
            annotation: filename of annotation file
        Returns:

        """
        if dir is not None:
            filepath = Path(dir) / Path(filename)
        else:
            filepath = Path(filename)
        signals = []
        times = []
        units = None
        headers = None
        with open(filepath, "rt", newline='') as csvfile:
            reader = csv.reader(csvfile, dialect=dialect)
            if header:
                row = next(reader)
                headers = row[1:] if fs is None else row
            for row in reader:
                if fs is None:
                    times.append(float(row[0]))
                    signals.append([float(x) for x in row[1:]])
                else:
                    signals.append([float(x) for x in row])
        if not transpose:
            data = np.array(signals).transpose()
        else:
            data = np.array(signals)
        chnum = data.shape[0]
        if units is None:
            units = [default_unit] * chnum
        if headers is None:
            headers = [f"C{i+1}" for i in range(chnum)]
        if fs is None:
            times = np.array(times)
            diff = times[1:] - times[:-1]
            fs = 1 / np.mean(diff)
            assert np.std(diff)*fs < 1e-4, "The time differences are not equal"
            assert all(diff > 0), "Some time steps are negative"
        c = SigContainer.from_signal_array(data, headers, units, float(fs))
        if annotation is not None:
            if isinstance(annotation, str):
                annotation = [annotation]
            for a in annotation:
                with open(a, "rt") as afile:
                    areader = csv.reader(afile)
                    samples = []
                    labels = []
                    notes = []
                    for time, label in areader:
                        samples.append(TimeUnit.to_sample(float(time), fs, TimeUnit.SECOND))
                        labels.append(label)
                        notes.append("")
            c.add_annotation(Path(a).stem, samples, labels, notes)
        return c

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


if __name__ == "__main__":
    p = DPath.from_path("a/x.y.z/x.y.z", dir=False)
    print(repr(p))
    print(str(p))
    p = p.extend_stem("00").resuffix(".png").prepend_path(DPath.from_path("/prep/p2", dir=True))
    print(repr(p))
    print(str(p))
    q = DPath.from_path("")
    q = q.prepend_path(DPath.from_path("", dir=True))
    q = q.base_path(DPath.from_path("x.y"))
    print(repr(q))
    print(str(q))
