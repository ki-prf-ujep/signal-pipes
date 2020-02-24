import configparser
import numpy as np
from sigpipes.sigcontainer import SigContainer
from typing import Iterable

class Synergy:
    def __init__(self, file_path: str, section:str):
        c = configparser.ConfigParser()
        with open(file_path, "rt") as f:
            contents = f.read()
        contents = contents.replace("/\n", ",")
        c.read_string(contents)
        self.fs = float(c[section]["Sampling Frequency(kHz)"].replace(",", ".")) * 1000
        data_key = [key for key in c[section].keys() if "data" in key][0]
        self.unit = float(c[section]["One ADC unit (µV)"].replace(",", "."))
        self.data = np.fromiter((self.unit * float(val) for val in c[section][data_key].split(",")),
                                   dtype=np.float64)
        self.label = section

    def sigcontainer(self) -> SigContainer:
        container = SigContainer.from_signal_array(self.data.reshape(1,2000),
                                                   channels=[self.label],
                                                   units=["µV"], fs=self.fs)
        return container

    @staticmethod
    def section_iter(file_path: str) -> Iterable[str]:
        c = configparser.ConfigParser()
        with open(file_path, "rt") as f:
            contents = f.read()
        contents = contents.replace("/\n", ",")
        c.read_string(contents)
        for section in c.keys():
            if "Data" in section:
                yield section