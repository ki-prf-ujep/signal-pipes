import numpy as np
from typing import Union, Sequence, Dict, Set, Any
import collections.abc

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

NON_THRESHOLD = {"IEMG", "MAV", "MMAV1", "MMAV2", "SSI", "VAR", "RMS", "WL", "LOG"}
WITH_THRESHOLD = {"WAMP", "SC", "ZC"}
ALL_SUPPORTED = NON_THRESHOLD | WITH_THRESHOLD

def features(data: np.ndarray, features: Set[str],
             thresholds: Dict[str, Union[float, Sequence[float]]] = None) -> Dict[str, np.ndarray]:
    r = {}
    n = data.shape[1]
    if {"IEMG", "MAV"} & features:
        absum = np.sum(np.abs(data), axis=1)
        if "IEMG" in features:
            r["IEMG"] = absum
        if "MAV" in features:
            r["MAV"] = absum / n

    if {"MMAV1", "MMAV2"} & features:
        data1 = np.abs(data[:, :n // 4])
        data2 = np.abs(data[:, n // 4:3 * n // 4])
        data3 = np.abs(data[:, 3 * n // 4:])
        wsum = np.sum(data2, axis=1)
        if "MMAV1" in features:
            r["MMAV1"] = (0.5 * np.sum(data1, axis=1) + wsum + 0.5 * np.sum(data3, axis=1)) / n
        if "MMAV2" in features:
            koef1 = 4 * np.arange(1, n // 4 + 1, dtype=np.float64) / n
            koef3 = 4 * (n - np.arange(3 * n // 4 + 1, n + 1, dtype=np.float64)) / n
            r["MMAV2"] = (np.sum(koef1 * data1, axis=1) + wsum + np.sum(koef3 * data3, axis=1)) / n

    if {"SSI", "VAR", "RMS"} & features:
        qsum = np.sum(data * data, axis=1)
        if "SSI" in features:
            r["SSI"] = qsum
        if "VAR" in features:
            r["VAR"] = qsum / (n - 1)
        if "RMS" in features:
            r["RMS"] = np.sqrt(qsum / n)

    if {"WL", "WAMP"} & features:
        df = np.abs(data[:, 1:] - data[:, :-1])
        if "WL" in features:
            r["WL"] = np.sum(df, axis=1)
        if "WAMP" in features:
            thresh = seq_wrap(thresholds["WAMP"])
            for t in thresh:
                r["WAMP({t})"] = np.sum(np.where(df >= t, 1, 0), axis=1)
    if "LOG" in features:
        r["LOG"] = np.exp(np.sum(np.log(np.abs(data)), axis=1) / n)
    if "SC" in features:
        thresh = seq_wrap(thresholds["SC"])
        for t in thresh:
            r[f"SC({t})"] = np.sum(np.where((data[:, 1:-1] - data[:, :-2]) * (data[:, 1:-1] - data[:, 2:])
                     >= t, 1, 0), axis=1)

    if "ZC" in features:
        for dif_thresh, mul_thresh in threshold["ZC"]:
            r[f"ZC({diff_thresh},{mul_thresh})"]  = np.sum(np.where(np.logical_and(data[:, :-1] * data[:, 1:] >= mul_thresh,
                                df >= diff_thresh), 1, 0), axis=1)
    return r


def feature(data: np.ndarray, feature: str, threshold: float = None) -> float:
    threshs = None
    if feature == "WAMP":
        threshs = {"WAMP": threshold}
    elif feature == "SC":
        threshs = {"SC" : threshold}
    return next(iter(features(data, {feature}, threshs).values()))[0]


if __name__ == "__main__":
    data = np.array([[1,2,3,4,5,6,7,8,9,10]])
    print(f'MMAV2={feature(data, "MMAV2")}')
