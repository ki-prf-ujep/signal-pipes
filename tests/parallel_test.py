from concurrent.futures import ProcessPoolExecutor as Executor
#from mpi4py.futures import MPIPoolExecutor as Executor

from sigpipes.sigfilter import Hilbert

from sigpipes.psigoperator import AsFuture, Barrier, ProcessTracker, FutureTracker
from sigpipes.sigoperator import Identity, RangeNormalization, Tee, Csv, Print, Fork, MVNormalization

from sigpipes.sources import SigGenerator, SignalType


if __name__ == "__main__":
    with Executor(4) as executor:
        if executor is not None:
            source = SigGenerator(100, SignalType.SINUS, 2, 5).sigcontainer()
            #future =  source | ProcessTracker() | AsFuture(executor) | Tee(RangeNormalization() | Csv(dir="/tmp") | ProcessTracker()) | Barrier() | ProcessTracker()

            futures = (source | ProcessTracker() | AsFuture(executor, id="start")
                              | Fork(RangeNormalization() | ProcessTracker(), MVNormalization() | ProcessTracker())
                              | ProcessTracker() | FutureTracker() | Barrier()
                              | ProcessTracker())
