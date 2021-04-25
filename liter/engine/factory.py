import inspect
from functools import partial
from .base import EngineBase
from .buffer import ScalarSmoother, to_buffer

__all__ = ["Automated"]


class Automated(EngineBase):
    def forward(self, batch, **kwargs):
        raise NotImplementedError("Method `forward` must be implemented.")

    @to_buffer("buffer_registry")
    def per_batch(self, batch, **kwargs):
        return self.forward(batch, **kwargs)

    def attach(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_forward(cls, func, smooth_window=50, **kwargs):
        assert inspect.isgeneratorfunction(
            func
        ), "The forward function must be a generator function with first arg being engine class placeholder."
        smooth_window = max(int(smooth_window), 1)
        buffer_names = _find_outputs(func)
        eng = cls()
        eng.forward = partial(func, eng)
        eng.attach(**{n: ScalarSmoother(smooth_window, **kwargs) for n in buffer_names})
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
