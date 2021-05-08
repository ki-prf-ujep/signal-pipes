import concurrent.futures as cf
from  dataclasses import dataclass


def _copy_future_state(source, destination):
    if source.cancelled():
        destination.cancel()
    if not destination.set_running_or_notify_cancel():
        return
    exception = source.exception()
    if exception is not None:
        destination.set_exception(exception)
    else:
        result = source.result()
        destination.set_result(result)


def _chain(pool, future, fn):
    result = cf.Future()

    def callback(_):
        try:
            temp = pool.submit(fn, future.result())
            copy = lambda _: _copy_future_state(temp, result)
            temp.add_done_callback(copy)
        except:
            result.cancel()
            raise

    future.add_done_callback(callback)
    return result


def _dchain(pool, futures, fn):
    result = cf.Future()
    bnum = len(futures)

    def callback(_):
        nonlocal bnum
        bnum -= 1
        if bnum > 0:
            return
        try:
            results = [f.result() for f in futures]
            temp = pool.submit(fn, *results)
            copy = lambda _: _copy_future_state(temp, result)
            temp.add_done_callback(copy)
        except:
            result.cancel()
            raise
    for f in futures:
        f.add_done_callback(callback)
    return result


def _identity(x):
    return x


class GNode:
    def __init__(self, context, *innodes):
        self.context = context
        self.innodes = innodes

    def __str__(self):
        return f"{self.context}{'['+','.join(str(n) for n in self.innodes) +']' if self.innodes else ''}"

@dataclass
class SignalSpace:
    fs: float
    sample_count: int
    channel_count: int
    lag: int


class SigFuture:
    def __init__(self, value, *, pool=None, fn=None, sigspace=None, node_description=None):
        self.sigspace = sigspace
        if isinstance(value, SigFuture):
            self.pool = value.pool
            self.future = _chain(self.pool, value.future, fn)
            self.depth = value.depth + 1
            self.node = GNode(node_description, value.node) if value.node is not None else None
        elif isinstance(value, tuple) and isinstance(value[0], SigFuture):
            self.pool = value[0].pool
            self.future = _dchain(self.pool, [f.future for f in value], fn)
            self.depth = max(f.depth for f in value) + 1
            prevnodes = [f.node for f in value]
            self.node = GNode(node_description, *value) if all(prevnodes) else None
        else:
            self.future = pool.submit(_identity, value)
            self.pool = pool
            self.depth = 0
            self.node = GNode(node_description) if node_description is not None else None

    def result(self):
        return self.future.result()

    def done(self):
        self.future.done()

    def __str__(self):
        return f"{self.sigspace}, depth: {self.depth}, graph: {self.node}"



