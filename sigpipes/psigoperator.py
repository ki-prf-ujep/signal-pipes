from concurrent.futures import Executor
from os import getpid
from typing import Any
from time import perf_counter

from sigpipes.sigcontainer import SigContainer, SigFuture
from sigpipes.sigoperator import Identity, SigOperator, ParallelSigOperator


class AsFuture(Identity):
    def __init__(self, executor: Executor):
        self.executor = executor

    def apply(self, container: SigContainer) -> Any:
        future = SigFuture(container, pool=self.executor)
        return future


class ProcessTracker(Identity):
    def __init__(self):
        pass

    def apply(self, container: SigContainer) -> Any:
        container = self.prepare_container(container)
        duration = perf_counter() % 1000.0
        print(f"{getpid()}\t[{duration:.4f}]:\t{container.id}")
        return container


class Barrier(Identity, ParallelSigOperator):
    def __init__(self):
        pass

    def apply(self, container: SigContainer) -> Any:
        raise Exception("Barrier is applicable only on futures")

    def par_apply(self, future: SigFuture):
        return future.result()
