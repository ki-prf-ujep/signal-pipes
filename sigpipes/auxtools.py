from enum import Enum
import numpy as np
import collections.abc
from typing import Sequence, Any, Iterable, Union


class TimeUnit(Enum):
    SAMPLE = 0
    SECOND = 1
    TIME_DELTA = 2

    def fix(self):
        return TimeUnit.SECOND if self == TimeUnit.TIME_DELTA else self

    @staticmethod
    def time_unit_mapper(value: Union[int, float, np.timedelta64]) -> "TimeUnit":
        """
        Mappping values of appropriate types (int, floats or Numpy timedeltas) to symbolic
        representation of types of time units
        Args:
            value: value of appropriate type

        Returns:
            symbols from `TimeUnit` enumeration
        """
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
        """
        Transformation of time intervals (= time sifts) in samples to others representation.

        Args:
            shift:  time interval in samples (integral value)
            fs:  sample frequency
            time_unit:  time representation
        """
        if time_unit == TimeUnit.SAMPLE:
            return shift
        elif time_unit == TimeUnit.SECOND:
            return int(shift * fs)
        elif time_unit == TimeUnit.TIME_DELTA:
            interval = int(1_000_000_000 / fs)
            return int(shift / np.timedelta64(interval, "ns"))


def common_value(iterable: Iterable[Any]) -> Any:
    """
    Function returns common value of iterable (all items are equal) or
    raises `ValurError` exception.

    Args:
        iterable: source of values

    Returns:
        one value
    """
    val = None
    for item in iterable:
        if val is None:
            val = item
        elif val != item:
            raise ValueError("Iterable does not contains common value")
    return val


def seq_wrap(x: Any) -> Sequence[Any]:
    """
    Function wraps scalar values to one-item sequences (sequences are returned
    unmodified)

    Args:
        x: scalar value or sequence

    Returns:
        sequence
    """
    if isinstance(x, collections.abc.Sequence):
        return x
    else:
        return (x,)


def type_info(obj: Any) -> str:
    """
    Extended type info for scalars, Python lists and Numpy ndarrays (for debug print)

    Args:
        obj: object of a supported type

    Returns:
        string with type information
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
    Smarter string representation of scalars, lists and arrays
    (for debug purposes)

    Args:
        obj: object of a supported type
        prefix: indention of parts of complex values

    Returns:
        string representation
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
    """
        Object with sequence interface (protocol) and infinite length based on
        unlimited repeating of finite pattern.
    """
    def __init__(self, iterator: Iterable[Any]):
        """
        Args:
            iterator:  recurring pattern (transformed to list before usage)
        """
        self.data = list(iterator)
        self.n = len(self.data)

    def __getitem__(self, item:int) -> Any:
        return self.data[item % self.n]

    def __str__(self) -> str:
        return str(self.data)

    def __repr__(self) -> str:
        return repr(self.data)

    def __len__(self) -> int:
        """
        Returns:
            This dunder method always raises exception (cyclic list has de facto infinite length)
        """
        raise NotImplementedError("Cyclic list has infinite length")


