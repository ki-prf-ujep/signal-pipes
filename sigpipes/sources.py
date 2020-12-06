import configparser
from collections import defaultdict

import numpy as np
from sigpipes.sigcontainer import SigContainer, DPath
from typing import Iterable
from re import match, finditer, Pattern
from pathlib import Path


class SynergyLP:
    def __init__(self, filename: str, *, dir: str = "", shortname = None, channels = None):
        self.filepath = DPath.from_path(filename).prepend_path(DPath.from_path(dir, dir=True))

        with open(str(self.filepath), "rt", encoding="utf-16le") as f:
            longline = None
            cline = False
            data = defaultdict(list)
            gsize = 0
            channel = 0
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

                m = match(r"Channel\s+Number=(\d+)", longline)
                if m:
                    channel = int(m.group(1))-1
                    continue

                m = match(r"(?:Sweep|LivePlay)\s+Data\(mV\)<(\d+)>=(.*)", longline)
                if m:
                    gsize += int(m.group(1))
                    dataline = m.group(2)
                    for subm in finditer(r"(-?)(\d+),(\d+),?", dataline):
                        value = int(subm.group(2)) + int(subm.group(3)) / 100
                        if subm.group(1) == "-":
                            value = -value
                        data[channel].append(value)
        self.data = data
        self.channels = channels

        if shortname is None:
            self.shortpath = self.filepath
        elif isinstance(shortname, Pattern):
            stem = self.filepath.stem
            m = shortname.search(stem)
            if m:
                stem = m.group(0)
            self.shortpath = self.filepath.restem(stem)

    def sigcontainer(self) -> SigContainer:
        if list(self.data.keys()) == [0]:
            data = np.array(self.data[0]).reshape((1, len(self.data[0])))
            container = SigContainer.from_signal_array(data, channels=[self.shortpath.stem], units=["mV"], fs=self.fs)
        else: # multichannel
            data = np.vstack(tuple(np.array(self.data[chan]).reshape(1, len(self.data[chan]))
                             for chan in sorted(self.data.keys())))
            if self.channels is None:
                labels = [f"{self.shortpath.stem}: channel {chan}" for chan in sorted(self.data.keys())]
            else:
                labels = [f"{self.shortpath.stem}: {self.channels[chan]}" for chan in sorted(self.data.keys())]

            container = SigContainer.from_signal_array(data, channels=labels, units=["mV"] * len(labels), fs=self.fs,
                                                       basepath=str(self.shortpath))
        return container
