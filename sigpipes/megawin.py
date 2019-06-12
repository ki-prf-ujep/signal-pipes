import numpy as np
from typing import Dict, Any
from sigpipes.sigcontainer import SigContainer


def matlab_hash_refine(h: Dict[str, Any]) -> Dict[str, Any]:
    nh = {}
    for key in h.keys():
        nh[key.strip("\x00")] = h[key]
    return nh


class MegaWinMatlab:
    def __init__(self, file_path: str):
        from scipy.io import loadmat
        self.m = matlab_hash_refine(loadmat(file_path, appendmat=True,
                                            squeeze_me=True, struct_as_record=False))
        self.data = self.m["datablock1"].data
        self.units = list(self.m["units"])
        self.fs = self.m["sampfreq"]
        self.channels = [f"{side}:{source}".replace(" ", "_")
                         for side, source in zip(self.m["sideinfo"], self.m["sources"])]
        self.annotations = {}
        if "markers" in self.m:
            self.register_annotation("markers")

    def register_annotation(self, annotator: str, symbol: str = '"') -> None:
        self.annotations[annotator] = {
            "sample":  (self.m[annotator] * self.fs).astype(np.int64),
            "symbol":  symbol
        }

    def write_to(self, record_name: str, dir_path: str, *, fmt: str = "32") -> None:
        """
        :param record_name:  name of record (i.e. of header and data file)
        :param dir_path: name of target directory
        :param fmt: physionet data format for digital signals (verified formats: 16, 32)
        """
        import wfdb
        wfdb.wrsamp(record_name, fs=self.fs, units=self.units, p_signal=self.data,
                    sig_name=self.channels, write_dir=dir_path, fmt=[fmt] * self.data.shape[1])
        for annotator in self.annotations.keys():
            sample = self.annotations[annotator]["sample"]
            symbols = [self.annotations[annotator]["symbol"]] * len(sample)
            wfdb.wrann(record_name, annotator, sample, symbol=symbols, write_dir=dir_path)

    def sigcontainer(self) -> SigContainer:
        container = SigContainer.from_signal_array(self.data.transpose(), channels=self.channels,
                                                   units=self.units, fs=self.fs)
        for annotator in self.annotations.keys():
            samples = self.annotations[annotator]["sample"]
            symbols = [self.annotations[annotator]["symbol"]] * len(samples)
            notes = "" * len(samples)
            container.add_annotation(annotator, samples, types=symbols, notes=notes)

        return container
