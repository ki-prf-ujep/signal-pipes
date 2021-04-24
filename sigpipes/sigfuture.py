import concurrent.futures as cf


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


class SigFuture:
    def __init__(self, value, *, pool=None, fn=None):
        if isinstance(value, SigFuture):
            self.pool = value.pool
            self.future = _chain(self.pool, value.future, fn)
        elif isinstance(value, tuple) and isinstance(value[0], SigFuture):
            self.pool = value[0].pool
            self.future = _dchain(self.pool, [f.future for f in value], fn)
        else:
            self.future = pool.submit(_identity, value)
            self.pool = pool

    def result(self):
        return self.future.result()

    def done(self):
        self.future.done()



