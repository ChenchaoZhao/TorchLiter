import collections
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from . import REPR_INDENT

__all__ = [
    "BufferBase",
    "SequenceContainer",
    "ExponentialMovingAverage",
    "ScalarSummaryStatistics",
    "ScalarSmoother",
    "VectorSmoother",
]


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


class SequenceContainer(BufferBase):
    """Sequence container Ingests new values and extends `self.value`"""

    values: List[Any]

    def reset(self):
        self.values = []

    def update(self, sequence: Union[List, Tuple]):
        s = list(sequence)
        self.values.extend(s)

    def state_dict(self):
        return dict(values=self.values)

    def load_state_dict(self, state_dict):
        self.values = list(state_dict["values"])

    def __len__(self) -> int:
        return len(self.values)


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
        alpha: float = 0.01,
        **kwargs: Any,
    ):
        assert 0 <= alpha <= 1, "Value `alpha` should be in [0, 1]."
        super().__init__(alpha=alpha, **kwargs)

    def reset(self):
        self.mean = 0.0
        self.variance = 0.0
        self._count = 0

    def update(self, x):
        delta = x - self.mean

        self.mean = self.mean + self.alpha * delta
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta**2)

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
        return self.variance**0.5


class _ScalarStatistics(BufferBase):
    """
    Base class for scalar statistics.

    The streaming scalars are stored in a queue
    of certain length (`maxlen`).
    If `maxlen` is not specified, then the queue
    is a list of any length.

    Available statistics:
        - mean
        - median
        - std
        - max
        - min
    """

    def __init__(self, maxlen: Optional[int] = None, **kwargs):
        if maxlen is not None:
            assert maxlen > 0, f"max_len should be positive but got {maxlen}"
            maxlen = max(1, int(maxlen))
        super().__init__(maxlen=maxlen, **kwargs)

    def reset(self):
        self._count = 0
        if self.maxlen is None:
            self._queue = []
        else:
            self._queue = collections.deque([], maxlen=self.maxlen)

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


class ScalarSummaryStatistics(_ScalarStatistics):
    """
    Store the scalars and compute statistics.

     The streaming scalars are stored in a list of
     any length. This is supposed to use in evals
     where the length is eval datasets.

    Available statistics:
        - mean
        - median
        - std
        - max
        - min
    """

    def __init__(self, **kwargs):
        super().__init__(maxlen=None, **kwargs)

    def __len__(self) -> int:
        return len(self._queue)


class ScalarSmoother(_ScalarStatistics):
    """
    Rolling average of a stream of scalars.

    The streaming scalars are stored in a deque
    of certain length (`maxlen`). The statistics
    are computed within the current deque.

    Available statistics:
        - mean
        - median
        - std
        - max
        - min
    """

    def __init__(self, window_size: int, **kwargs):
        window_size = int(window_size)
        assert window_size > 0, f"window_size should be > 0 but get {window_size}"
        super().__init__(maxlen=window_size, **kwargs)


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
