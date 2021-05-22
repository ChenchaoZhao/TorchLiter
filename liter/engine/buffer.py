import abc
import collections
from functools import wraps

import numpy as np
import torch
import torch.nn.functional as F

from . import REPR_INDENT

__all__ = ["to_buffer", "BufferBase", "ScalarSmoother"]


def to_buffer(name="buffer_registry"):
    # name should be an attribute of the owner class

    def decorator(func):
        # func: class method that yields tuple of (key: str, val)
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            buffer_dict = getattr(self, name)
            for key, val in func(self, *args, **kwargs):
                if key in buffer_dict:
                    buffer_dict[key](val)  # pushing update by `__call__`

        return wrapper

    return decorator


class BufferBase(abc.ABC):
    """Buffer base class."""

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "There should not be any args only kwargs allowed."
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.reset()

    @abc.abstractmethod
    def update(self, x):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def state_dict(self):
        pass

    @abc.abstractmethod
    def load_state_dict(self, state_dict):
        pass

    def __call__(self, x):
        self.update(x)

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            out.append(" " * REPR_INDENT + f"{k}: {v}")
        return "\n".join(out)


class ScalarSmoother(BufferBase):
    """Rolling smoothing buffer for scalars."""

    def __init__(self, window_size: int, **kwargs):

        window_size = int(window_size)
        assert window_size > 0, f"window_size should be > 0 but get {window_size}"

        super().__init__(window_size=window_size, **kwargs)

    def reset(self):
        self._count = 0
        self._queue = collections.deque([], maxlen=self.window_size)

    def update(self, x: float):
        self._queue.append(x)
        self._count += 1

    def state_dict(self):
        return {"queue": self._queue, "count": self._count}

    def load_state_dict(self, state_dict):
        self._count = state_dict["count"]
        self._queue = state_dict["queue"]

    @property
    def mean(self):
        return np.mean(self._queue) if len(self._queue) > 0 else 0.0

    @property
    def median(self):
        return np.median(self._queue) if len(self._queue) > 0 else 0.0

    @property
    def std(self):
        return np.std(self._queue) if len(self._queue) > 0 else 0.0

    @property
    def max(self):
        return np.max(self._queue) if len(self._queue) > 0 else 0.0

    @property
    def min(self):
        return np.min(self._queue) if len(self._queue) > 0 else 0.0


class VectorSmoother(BufferBase):
    def __init__(
        self, alpha: float, n_dim: int, init_value: float, eps: float = 1e-8, **kwargs
    ):
        alpha = float(alpha)
        n_dim = max(1, int(n_dim))
        assert (
            alpha >= 0 and alpha <= 1
        ), f"Parameter alpha = {alpha} should be in [0, 1]"
        super().__init__(
            alpha=alpha, n_dim=n_dim, init_value=init_value, eps=eps, **kwargs
        )

    def reset(self):
        self._count = 0
        self._state = torch.zeros(n_dim, dtype=float) + self.init_value

    def update(self, x: torch.Tensor):
        self._state = self.alpha * x + (1 - self.alpha) * self._state

    def state_dict(self):
        return {"vector": self._state, "count": self._count}

    def load_state_dict(self, state_dict):
        self._count = state_dict["count"]
        self._state = state_dict["vector"]

    @property
    def raw_vector(self):
        return self._state

    @property
    def l1_normalized(self):
        return F.normalize(self._state, dim=0, p=1.0)

    @property
    def l2_normalized(self):
        return F.normalize(self._state, dim=0, p=2.0)

    def lp_normalized(self, p: float):
        return F.normalize(self._state, dim=0, p=p)
