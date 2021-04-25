import inspect
from .base import EngineBase
from .buffer import ScalarSmoother, to_buffer

__all__ = ["Automated", "to_engine"]


class Automated(EngineBase):
    def forward(self, batch, **kwargs):
        raise NotImplementedError("Method `forward` must be implemented.")

    @to_buffer("buffer_registry")
    def per_batch(self, batch, **kwargs):
        self.forward(batch, **kwargs)

    def attach(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_forward(cls, func):
        assert inspect.isgeneratorfunction(
            func
        ), "The forward function must be a generator function."
        cls.forward = func
        return cls


def to_engine(smooth_window=50, **kwargs):

    smooth_window = max(int(smooth_window), 1)

    def decorator(func):

        assert inspect.isgeneratorfunction(
            func
        ), "The forward function must be a generator function."

        buffer_names = _find_outputs(func)

        eng = Automated.from_forward(func)
        eng.attach(**{n: ScalarSmoother(smooth_window, **kwargs) for n in buffer_names})

        return eng

    return decorator


def _find_outputs(func):
    source = inspect.getsource(func)
    names = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("yield"):
            line = line.replace("yield", "")
            names.append(line.split(",")[0].strip()[1:-1])
    return names
