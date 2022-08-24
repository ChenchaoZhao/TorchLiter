"""Exceptions used by the `Engine` class."""


class BreakIteration(Exception):
    """
    BreakIteration Exception.

    Parameters
    ----------
    shutdown_engine : bool
        If flag variable `shutdown_engine` (the default is False) is `True`,
        the `Engine` object will kill both the interation and the epochs

    Attributes
    ----------
    shutdown_engine
    """

    def __init__(self, shutdown_engine: bool = False):
        self.shutdown_engine = shutdown_engine


class ContinueIteration(Exception):
    """
    ContinueIteration Exception.

    When raised, the `Engine` iteration will continue.
    """


class BadBatchError(ContinueIteration):
    """
    BadBatchError Exception, subclass of ContinueIteration.

    If current batch of data is corrupted, skip current batch and fetch a new
    batch and then continue.
    """


class StopEngine(BreakIteration):
    """
    StopEngine Exception, subclass of BreakIteration.

    When raised at batch iteration level, if `shutdown_engine=True` then reraise
    BreakIteration to terminate engine.
    """

    def __init__(self):
        super().__init__(True)


class GradientExplosionError(StopEngine):
    """GradientExplosionError, subclass of StopEngine."""


class FoundNanError(StopEngine):
    """FoundNanError, subclass of StopEngine."""


class EarlyStopping(StopEngine):
    """EarlyStopping, subclass of StopEngine."""
