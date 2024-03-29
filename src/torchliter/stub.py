from typing import *

from . import REPR_INDENT
from .utils import _convert_str_to_py_object_name

__all__ = ["StubBase", "Train", "Evaluate", "Lambda"]


class StubBase:
    """Base class for Stubs."""

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, "There should not be any args only kwargs allowed."
        for k, v in kwargs.items():
            setattr(self, k, v)

    def replicate(self, copy: int = 1):
        # make copies of stubs with same kwargs
        copy = int(copy)
        assert copy > 0
        kwargs = self.__dict__
        copies = []
        for _ in range(copy):
            copies.append(self.__class__(**kwargs))
        return copies

    def __call__(self, copy: int = 1) -> Optional[List["StubBase"]]:
        """
        Call method.

        Parameters
        ----------
        copy : int
            The number of copies of `Stub` (the default is 1).

        Returns
        -------
        Optional[List["StubBase"]]
            If copy is 0 return None
            If copy > 0, return a list of `StubBase`
        """

        if copy > 0:
            return self.replicate(copy)
        else:
            return None

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            out.append(" " * REPR_INDENT + f"{k}: {v}")

        return "\n".join(out)


class Train(StubBase):
    """Train stub."""

    def __init__(self, dataloader: str, **kwargs):

        assert isinstance(
            dataloader, str
        ), f"Dataloader should be string type but get type {type(dataloader)}"

        kwargs["dataloader"] = dataloader
        kwargs["action"] = "train"
        kwargs["epoch"] = 1
        kwargs["iteration"] = 0
        # additional options can be `optimizer`, `scheduler`

        super().__init__(**kwargs)


class Evaluate(StubBase):
    """Evaluation stub."""

    def __init__(self, dataloader: str, **kwargs):

        assert isinstance(
            dataloader, str
        ), f"Dataloader should be string type but get type {type(dataloader)}"
        kwargs["dataloader"] = dataloader
        kwargs["action"] = "evaluate"
        kwargs["epoch"] = 1
        kwargs["iteration"] = 0
        # additional options can be metrics and eval specs

        super().__init__(**kwargs)


class Lambda(StubBase):
    """General action stub."""

    def __init__(self, action: str, **kwargs):

        assert isinstance(
            action, str
        ), f"Action should be string type but get type {type(action)}."

        assert action != "train", "Please use Train stub for training action."
        assert action != "evaluate", "Please use Evaluate stub for eval action."

        kwargs["action"] = _convert_str_to_py_object_name(action)
        # e.g. save checkpoint, send emails, etc.

        super().__init__(**kwargs)
