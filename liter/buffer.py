import abc
import collections

import numpy as np

from .common import REPR_INDENT


class BufferBase(abc.ABC):
    """Buffer base class"""
    def __init__(self, *args, **kwargs):
        assert len(args) == 0, \
        "There should not be any args only kwargs allowed."
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.reset()

    @abc.abstractmethod
    def update(self, x):
        pass

    @abc.abstractmethod
    def reset(self):
        pass

    def __call__(self, x):
        self.update(x)

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            out.append(' ' * REPR_INDENT + f"{k}: {v}")
        return '\n'.join(out)


class ScalarSmoother(BufferBase):
    """Rolling smoothing buffer for scalars"""
    def __init__(self, window_size: int, **kwargs):

        window_size = int(window_size)
        assert window_size > 0, \
        f"window_size should be > 0 but get {window_size}"

        super().__init__(window_size=window_size, **kwargs)

    def reset(self):
        self._count = 0
        self._queue = collections.deque([], maxlen=self.window_size)

    def update(self, x: float):
        self._queue.append(x)
        self._count += 1

    @property
    def mean(self):
        return np.mean(self._queue)

    @property
    def median(self):
        return np.median(self._queue)

    @property
    def std(self):
        return np.std(self._queue)

    @property
    def max(self):
        return np.max(self._queue)

    @property
    def min(self):
        return np.min(self._queue)
