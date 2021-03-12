import configparser
import abc
from collections import defaultdict
from enum import Enum
from re import match, finditer, Pattern
from typing import Iterable

import numpy as np
import scipy.signal as signal

from sigpipes.joiner import Merge
from sigpipes.sigcontainer import SigContainer, DPath


class SignalSource(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def sigcontainer(self) -> SigContainer:
        raise NotImplementedError("abstract method")

    @property
    @abc.abstractmethod
    def filepath(self) -> str:
        raise NotImplementedError("abstract method")


class SynergyLP(SignalSource):
    def __init__(self, filename: str, *, dir: str = "", shortname = None, channels = None):
        self._filepath = DPath.from_path(filename).prepend_path(DPath.from_path(dir, dir=True))

        with open(str(self._filepath), "rt", encoding="utf-16le") as f:
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
            self.shortpath = self._filepath
        elif isinstance(shortname, Pattern):
            stem = self._filepath.stem
            m = shortname.search(stem)
            if m:
                stem = m.group(0)
            self.shortpath = self._filepath.restem(stem)

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

    @property
    def filepath(self) -> str:
        return str(self.shortpath)


class SignalType(Enum):
    SINUS = 'Sinus'
    SAWTOOTH = 'Sawtooth'
    SQUARE = 'Square'
    TRIANGLE = 'Triangle'


class NoiseType(Enum):
    WHITE = 'White'
    PINK = 'Pink'
    RED = 'Red'


class SigGenerator(SignalSource):
    """
       Used for generating waveforms of different types.

       Parameters
       ----------
           sig_type: [SignalType]
                    What type of a waveform will be generated.
                    SINUS: Sine-wave waveform.
                    SAWTOOTH: Sawtooth-wave waveform.
                    SQUARE: Square-wave waveform.
                    TRIANGLE: Triangle-wave waveform.

           fs: [Float]
               Sampling frequency in [Hz].

           sig_duration: [Float]
                        The duration of generated waveform in [s].

           sig_frequency: [Float]
                         frequency of the waveform in [Hz].

           sig_amplitude: [Float]
                         Height of the waveform in user defined units.

           sig_phase: [Float]
                     Phase shift of the wave form: 0 - 2 π.

           amplitude_units: [List]
                           User defined name of the y-axis. Default is ''.

           channel_name: [List]
                        User defined name of the channel. Default is the
                        chosen waveform type, frequency, phase, amplitude
                        and sample rate.
           """

    def __init__(self, fs: float, sig_type: SignalType, sig_duration: float, sig_frequency: float,
                 sig_amplitude: float = 1.0, sig_phase: float = 0.0, *, amplitude_units=None, channel_name=None):
        self.sig_type = sig_type
        self.fs = fs
        self.sig_duration = sig_duration
        self.sig_frequency = sig_frequency
        self.sig_amplitude = sig_amplitude
        self.sig_phase = sig_phase
        self.amplitude_units = amplitude_units
        self.channel_name = channel_name

    @staticmethod
    def __sig_switch(sig_type, sig_frequency, sig_amplitude, sig_phase, x):
        """
        Used for switching the type of generated Waveform.
        """

        return {
            'Sinus': sig_amplitude * np.sin((2 * np.pi * sig_frequency * x) + (sig_phase * np.pi)),
            'Sawtooth': sig_amplitude * signal.sawtooth((2 * np.pi * sig_frequency * x) + (sig_phase * np.pi)),
            'Square': sig_amplitude * signal.square((2 * np.pi * sig_frequency * x) + (sig_phase * np.pi)),
            'Triangle': sig_amplitude * signal.sawtooth((2 * np.pi * sig_frequency * x) + (sig_phase * np.pi),
                                                        width=0.5) * sig_amplitude
        }[sig_type.value]

    def sigcontainer(self) -> SigContainer:
        """
        Returns
        -------
        SigContainer
            Generated waveform in a SigContainer type.

        """
        if self.channel_name is None:
            self.channel_name = [
                self.sig_type.value + '(f=' + str(self.sig_frequency) + ', φ='
                + str(self.sig_phase) + 'π, A=' + str(self.sig_amplitude)]
        if self.amplitude_units is None:
            self.amplitude_units = ['']

        x = np.linspace(0, self.sig_duration, int(self.fs * self.sig_duration)).reshape(1, int(
            self.fs * self.sig_duration))
        sig_base = self.__sig_switch(self.sig_type, self.sig_frequency, self.sig_amplitude, self.sig_phase, x)
        generated_sig = SigContainer.from_signal_array(sig_base, self.channel_name, self.amplitude_units,
                                                       float(self.fs), basepath=self.filepath)
        return generated_sig

    @property
    def filepath(self):
        return f"{self.sig_type.value}_{self.sig_frequency}_{self.sig_phase}_{self.sig_amplitude}"


class NoiseGenerator(SignalSource):
    """
        Used for generating noise of different types. Returns SigContainer
        of the chosen noise.

        Parameters
        ----------
            noise_type: [NoiseType]
                     What type of a noise will be generated.
                     WHITE: Sine-wave waveform.
                     PINK: Sawtooth-wave waveform.
                     RED: Square-wave waveform.

            st_deviation: [Float]
                         Standard deviation.

            noise_duration: [Float]
                           The duration of generated noise in [s].

            fs: [Float]
                Sampling frequency in [Hz].

            channels: [Int]
                      How many channels of noise will be created.

            amplitude_units: [List]
                            User defined name of the y-axis. Default is empty
                            string.

            channel_names: [List]
                         User defined name of the channel. Default is the
                         channel number and standard deviation.
        """

    def __init__(self, fs: float, noise_type: NoiseType, noise_duration: float, st_deviation: float = 1.0, *,
                 channels: int = 1, channel_names: list = None, amplitude_units: list = None):
        self.noise_type = noise_type
        self.st_deviation = st_deviation
        self.noise_duration = noise_duration
        self.fs = fs
        self.channels = channels
        self.channel_names = channel_names
        self.amplitude_units = amplitude_units

    @staticmethod
    def __pink_red_noise(noise_type, fs, st_deviation):
        """
        Used for generating pink or red noise

        Based on the algorithm in:
        Timmer, J. and Koenig, M.:
        On generating power law noise.
        Astronomy and Astrophysics, v.300, p.707-710 (1995)

        Available from URL:
        http://articles.adsabs.harvard.edu//full/1995A%26A...300..707T/0000708.000.html

        Parameters:
        --------------
        noise_type:[Int]
                  1: Pink noise
                  2: Red noise
        """

        if noise_type == 1:
            # Pink noise
            exponent = 1
        else:
            # Red noise
            exponent = 2

        min_freq = 1 / fs
        frequency = np.fft.rfftfreq(int(fs))
        sig_scale = frequency

        x = np.sum(sig_scale < min_freq)
        if x < len(sig_scale):
            sig_scale[:x] = sig_scale[x]
            sig_scale = sig_scale ** (-exponent / 2)

        sig_phase = np.random.normal(scale=sig_scale, size=len(frequency))
        sig_power = np.random.normal(scale=sig_scale, size=len(frequency))
        noise = (np.fft.irfft(sig_power + sig_phase * 1J, n=int(fs)) * st_deviation)
        return noise

    def __noise_switch(self, noise_type, st_deviation, noise_duration, fs):
        """
        Used for switching the type of generated noise.
        """

        return {
            'White': np.random.normal(0, st_deviation, int(fs * noise_duration)),
            'Pink': self.__pink_red_noise(1, int(fs * noise_duration), st_deviation),
            'Red': self.__pink_red_noise(2, int(fs * noise_duration), st_deviation)
        }[noise_type.value]

    def sigcontainer(self) -> SigContainer:
        """
        Returns
        -------
        SigContainer
            Generated noise in a SigContainer type.

        """
        noise = self.__noise_switch(self.noise_type, self.st_deviation, self.noise_duration, self.fs) \
            .reshape(1, int(self.fs * self.noise_duration))

        for _ in range(self.channels - 1):
            noise = np.vstack(
                    [noise, self.__noise_switch(self.noise_type, self.st_deviation, self.noise_duration, self.fs)])
        if self.channel_names is None:
            self.channel_names = [f"{self.noise_type.value} noise (σ={self.st_deviation})"] * self.channels
        if self.amplitude_units is None:
            self.amplitude_units = [' '] * self.channels

        generated_noise = SigContainer.from_signal_array(noise, self.channel_names, self.amplitude_units,
                                                         float(self.fs), basepath=self.filepath)
        return generated_noise

    @property
    def filepath(self):
        return f"{self.noise_type.value}_noise_{self.st_deviation}"


class ConstantSignal(SignalSource):
    def __init__(self, fs: float, duration: float, value: float, channels: int = 1,
                 *, channel_names: Iterable[str] = None, units: Iterable[str] = None):
        self.fs = fs
        self.duration = duration
        self.value = value
        self.channels = channels
        self.channel_names = channel_names
        self.units = units

    def sigcontainer(self) -> SigContainer:
        data = np.full((self.channels, int(self.duration * self.fs)), self.value, dtype=np.float64)
        if self.channel_names is None:
            self.channel_names = [f"constant:{self.value}"] * self.channels
        if self.units is None:
            self.units = [''] * self.channels

        return SigContainer.from_signal_array(data, self.channel_names, self.units, self.fs, basepath=self.filepath)

    @property
    def filepath(self) -> str:
        return f"const_{self.value}"


if __name__ == "__main__":
    from sigpipes.plotting import Plot
    from sigpipes.sigoperator import Print
    (SigGenerator(100, SignalType.SAWTOOTH, 2.0, 10).sigcontainer()
         | Plot()
         | Merge(np.add, NoiseGenerator(100, NoiseType.PINK, 2.0, 0.5).sigcontainer() | Plot())
         | Merge(np.maximum, ConstantSignal(100, 2.0, 0.0).sigcontainer())
         | Plot()
    )