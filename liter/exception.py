class BreakIteration(Exception):
    def __init__(self, shutdown_engine=False):
        self.shutdown_engine = shutdown_engine


class ContinueIteration(Exception):
    pass


class BadBatchError(ContinueIteration):
    """
    Use case:

    current batch of data is corrupted, skip current batch and fetch a new batch
    and then continue.
    """


class StopEngine(BreakIteration):
    """
    Use case:

    when raised at batch iteration level, if `shutdown_engine=True` then reraise
    BreakIteration to terminate engine.
    """

    def __init__(self):
        super().__init__(True)


class GradientExplosionError(StopEngine):
    pass


class FoundNanError(StopEngine):
    pass


class EarlyStopping(StopEngine):
    pass
