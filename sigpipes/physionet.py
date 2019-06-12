import wfdb
import numpy as np
import h5py
import pickle
from typing import List, Iterable, Dict, Any
from sigpipes.sigcontainer import SigContainer


def dumpa(obj: Any) -> np.ndarray:
    """
    Pickling to numpy array (which are serializable to hdf5 files)
    :param obj: pickled object
    :return: numpy byte array
    """
    d = pickle.dumps(obj)
    return np.frombuffer(d, dtype=np.uint8)


class PhysionetRecord:
    def __init__(self, record_name: str, database: str = None):
        self.name = record_name
        self.database = database
        self.path = f"{self.database}/{self.name}"
        rcpath = f"{self.path}/_record"
        with h5py.File("physionet_cache.h5") as store:
            if rcpath not in store:
                self.record = wfdb.rdrecord(record_name, pb_dir=database)
                store.create_dataset(rcpath, data=dumpa(self.record), compression="gzip")
            else:
                self.record = pickle.loads(store[rcpath][:])
        self.annotations = {}

    @property
    def units(self) -> List[str]:
        return self.record.units

    def sigcontainer(self, annotators: Iterable[str] = None) -> SigContainer:
        c = SigContainer.from_signal_array(signals=np.transpose(self.record.p_signal),
                                           channels=self.record.sig_name,
                                           units=self.record.units, fs=self.record.fs)
        if annotators is not None:
            with h5py.File("physionet_cache.h5") as store:
                for annotator in annotators:
                    if annotator not in self.annotations:
                        annopath = f"{self.path}/{annotator}"
                        if annopath not in store:
                            self.annotations[annotator] = wfdb.rdann(self.name, annotator,
                                                                     pb_dir=self.database)
                            store.create_dataset(annopath, data=dumpa(self.annotations[annotator]),
                                                 compression="gzip")
                        else:
                            self.annotations[annotator] = pickle.loads(store[annopath][:])
                    data = self.annotations[annotator]
                    c.add_annotation(annotator, data.sample, data.symbol, data.aux_note)
        return c



