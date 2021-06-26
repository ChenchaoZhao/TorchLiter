import collections
from functools import wraps
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import REPR_INDENT

__all__ = [
    "to_buffer",
    "BufferBase",
    "ExponentialMovingAverage",
    "ScalarSmoother",
    "VectorSmoother",
]


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


class BufferBase:
    """Buffer base class."""

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "There should not be any args only kwargs allowed."
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.reset()

    def update(self, x: Any):
        raise NotImplementedError("Method `update` must be implemented.")

    def reset(self):
        raise NotImplementedError("Method `reset` must be implemented.")

    def state_dict(self):
        raise NotImplementedError("Method `state_dict` must be implemented.")

    def load_state_dict(self, state_dict):
        raise NotImplementedError("Method `load_state_dict` must be implemented.")

    def __call__(self, x: Any):
        self.update(x)

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            out.append(" " * REPR_INDENT + f"{k}: {v}")
        return "\n".join(out)


class ExponentialMovingAverage(BufferBase):
    """
    Exponential Moving Average of a series of Tensors.

    update rule:
        EMA[x[t]] := (1 - alpha) * EMA[x[t-1]] + alpha * x[t]
    diff:
        delta[x[t]] := x[t] - EMA[x[t-1]]
    """

    mean: Union[float, Tensor]
    variance: Union[float, Tensor]

    def __init__(
        self,
        alpha: Optional[float] = None,
        window_size: Optional[int] = None,
        **kwargs: Any,
    ):
        if alpha is None:
            assert (
                window_size is not None
            ), "Init args `alpha` and `window_size` cannot be both `None`."
            alpha = 1.0 / window_size
        assert 0 <= alpha <= 1, "Value `alpha` should be in [0, 1]."
        super().__init__(alpha=alpha, **kwargs)

    def reset(self):
        self.mean = 0.0
        self.variance = 0.0
        self._count = 0

    def update(self, x):
        delta = x - self.mean

        self.mean = self.mean + self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta ** 2)

        self._count += 1
        self._delta = delta

    def state_dict(self):
        return {"count": self._count, "mean": self.mean, "variance": self.variance}

    def load_state_dict(self, state_dict):
        self._count = max(state_dict["count"], 0)
        self.mean = state_dict["mean"]
        self.variance = state_dict["variance"]

    @property
    def std(self):
        return self.variance ** 0.5


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


class VectorSmoother(ExponentialMovingAverage):
    """
    Exponential moving average of n-dim vector:

    vector = alpha * new_vector + (1 - alpha) * vector

    Additional features:
        l_p normalization
    """

    def __init__(
        self,
        alpha: float,
        n_dim: int,
        init_value: float,
        eps: float = 1e-8,
        normalize: bool = True,
        p: float = 1.0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        alpha = float(alpha)
        n_dim = max(1, int(n_dim))

        if dtype in (torch.int, torch.long):
            assert (
                not normalize
            ), "If dtype is `int` or `long`, normalize must be `False`. "
        super().__init__(
            alpha=alpha,
            n_dim=n_dim,
            init_value=init_value,
            eps=eps,
            normalize=normalize,
            p=p,
            device=torch.device(device),
            dtype=dtype,
            **kwargs,
        )

    def reset(self):
        self._count = 0
        self.mean = torch.zeros(self.n_dim) + self.init_value
        self.mean = self.mean.to(device=self.device, dtype=self.dtype)
        self.variance = torch.zeros_like(self.mean)
        if self.normalize:
            self.mean = self.lp_normalized(self.p)

    def update(self, x: torch.Tensor):
        super().update(x)
        if self.normalize:
            self.mean = self.lp_normalized(self.p)

    def state_dict(self):
        return {"mean": self.mean, "variance": self.variance, "count": self._count}

    def load_state_dict(self, state_dict):
        self._count = state_dict["count"]
        self.mean = state_dict["mean"]
        self.variance = state_dict["variance"]

    @property
    def vector(self):
        return self.mean

    @property
    def l1_normalized(self):
        return self.lp_normalized(1.0)

    @property
    def l2_normalized(self):
        return self.lp_normalized(2.0)

    @property
    def l1_norm(self):
        return self.lp_norm(1.0)

    @property
    def l2_norm(self):
        return self.lp_norm(2.0)

    def lp_norm(self, p: float):
        return self.mean.norm(dim=0, p=p)

    def lp_normalized(self, p: float):
        return F.normalize(self.mean, dim=0, p=p, eps=self.eps)
