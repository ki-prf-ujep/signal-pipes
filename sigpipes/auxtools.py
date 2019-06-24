from enum import Enum
import numpy as np
import collections.abc
from typing import Sequence, Any, Iterable


class TimeUnit(Enum):
    SAMPLE = 0
    SECOND = 1
    TIME_DELTA = 2

    def fix(self):
        return TimeUnit.SECOND if self == TimeUnit.TIME_DELTA else self


def common_value(iterable: Iterable[Any]) -> Any:
    val = None
    for item in iterable:
        if val is None:
            val = item
        elif val != item:
            raise ValueError("Iterable does not contains common value")
    return val


def seq_wrap(x: Any) -> Sequence[Any]:
    if isinstance(x, collections.abc.Sequence):
        return x
    else:
        return (x,)


def smart_copy(obj: Any, indices: Sequence[int] = None) -> Any:
    """
    Shallow copy of scalars, lists and arrays.
    """
    if isinstance(obj, (int, str, float)):
        return obj
    elif isinstance(obj, collections.abc.Sequence):
        if indices is None:
            return [smart_copy(item) for item in obj]
        else:
            return [smart_copy(obj[i]) for i in indices]
    elif isinstance(obj, np.ndarray):
        if indices is None:
            return obj.copy()
        else:
            return obj[indices]
    else:
        raise TypeError("Unsupported type")


def type_info(obj: Any) -> str:
    """
    Extended type info for scalars, Python lists and Numpy ndarrays
    :param obj: object of a supported type
    :return: string with type information
    """
    if isinstance(obj, (int, float, str)):
        return obj.__class__.__name__
    elif isinstance(obj, collections.abc.Sequence):
        return f"{obj.__class__.__name__}({len(obj)})"
    elif isinstance(obj, np.ndarray):
        return f"ndarray{obj.shape}"
    else:
        raise TypeError("Unsupported type")


def smart_tostring(obj: Any, prefix: int = 0) -> str:
    """
    Smarter brief string representation of scalars, lists and arrays
    (for debug purposes)
    :param obj: object of a supported type
    :param prefix: indention of parts of complex values
    :return: string representation
    """
    if isinstance(obj, (int, float, str, collections.abc.Sequence)):
        return str(obj)
    elif isinstance(obj, np.ndarray):
        return ("\n" + " "*prefix +
                np.array2string(obj, precision=3, suppress_small=True, threshold=8,
                                prefix=" "*prefix,
                                edgeitems=3, floatmode='maxprec'))
    else:
        raise TypeError("Unsupported type")


class CyclicList(collections.abc.Sequence):
    def __init__(self, iterator: Iterable[Any]):
        self.data = list(iterator)
        self.n = len(self.data)

    def __getitem__(self, item:int) -> Any:
        return self.data[item % self.n]

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def __len__(self) -> int:
        raise NotImplementedError("Cyclic list has infinite length")


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
