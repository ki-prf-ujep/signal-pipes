import configparser
import numpy as np
from sigpipes.sigcontainer import SigContainer
from typing import Iterable
from re import match, finditer
from pathlib import Path

class SynergyLP:
    def __init__(self, filename: str, *, dir:str = None):
        if dir is None:
            file = Path(filename)
        else:
            file = Path(dir) / Path(filename)
        with open(file, "rt", encoding="utf-16le") as f:
            longline = None
            cline = False
            data = []
            gsize = 0
            for line in f:
                line = line.rstrip()
                if cline:
                    longline += line.rstrip("/")
                else:
                    longline = line.rstrip("/")
                if line.endswith("/"):
                    longline += ","
                    cline = True
                    continue
                else:
                    cline = False

                m = match(r"Sampling Frequency\(kHz\)=(\d+,\d+)", longline)
                if m:
                    self.fs = 1000 * float(m.group(1).replace(",","."))
                    continue

                m = match(r"LivePlay Data\(mV\)<(\d+)>=(.*)", longline)
                if m:
                    gsize += int(m.group(1))
                    dataline = m.group(2)
                    for subm in finditer(r"(-?)(\d+),(\d+),?", dataline):
                        value = int(subm.group(2)) + int(subm.group(3)) / 100
                        if subm.group(1) == "-":
                            value = -value
                        data.append(value)
        self.data = np.array(data)
        self.label = file.stem
        self.fs = 50000

    def sigcontainer(self) -> SigContainer:
        container = SigContainer.from_signal_array(self.data.reshape((1,len(self.data))),
                                                   channels=[self.label],
                                                   units=["µV"], fs=self.fs)
        return container


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