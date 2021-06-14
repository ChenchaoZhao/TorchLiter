import inspect
from functools import partial

from .base import EngineBase
from .buffer import ScalarSmoother, to_buffer

__all__ = ["Automated"]


class Automated(EngineBase):
    """
    Automated Engine Given a forward generator function, `from_forward` will
    return an Automated engine class.

    For example
    ==================

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    @Automated.config(smooth_window=100)
    def classification(engine, batch):
        # the first arg must be a place holder for engine class

        engine.train()
        x, y = batch
        lgs = engine.model(x)
        loss = F.cross_entropy(lgs, y)

        yield "loss", loss.item()
        # metrics will be registered as buffers

        acc = (lgs.max(-1).indices == y).float().mean()

        yield "acc", acc.item()

    # or alternatively
    # `from_forward` will be deprecated
    classification = Automated.from_forward(classification)

    # attach other components such as model, optimizer, dataloader, etc.
    eng.attach(model=nn.Linear(2, 2))
    ...
    """

    def __init__(self, core_function, smooth_windown=50, **kwargs):
        assert inspect.isgeneratorfunction(core_function), (
            "The forward function must be a generator function "
            "with first arg being engine class placeholder."
        )
        super().__init__()
        self.core = partial(core_function, self)
        smooth_window = max(int(smooth_windown), 1)
        buffer_names = _find_outputs(core_function)
        self.attach(
            **{n: ScalarSmoother(smooth_window, **kwargs) for n in buffer_names}
        )

    @classmethod
    def config(cls, **kwargs):
        """
        Used as decorator for core function allowing user to attach additional
        init keyword args Examples.

        @Automated
        def core_func(ng, batch):
            ...


        @Automated.config(smooth_window=100)
        def core_func(ng, batch):
            ...
        """
        return partial(cls, **kwargs)

    def core(self, batch, **kwargs):
        raise NotImplementedError("Method `core` must be implemented.")

    @to_buffer("buffer_registry")
    def per_batch(self, batch, **kwargs):
        return self.core(batch, **kwargs)

    def attach(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_forward(cls, func, smooth_window=50, **kwargs):
        """
        This method is deprecated.

        Use init as decorator or cls.config(...) as decorator
        """
        eng = cls(func, smooth_window)
        return eng


def _find_outputs(func):
    source = inspect.getsource(func)
    names = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("yield"):
            line = line.replace("yield", "")
            names.append(line.split(",")[0].strip()[1:-1])
    return names
