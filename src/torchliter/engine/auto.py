import warnings
from inspect import isfunction, isgeneratorfunction
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Type, Union

from .. import REPR_INDENT, stub
from ..utils import _convert_str_to_py_object_name as _py_name
from .buffers import BufferBase
from .events import Engine, EventHandler
from .utils import _find_output_names, to_buffer

__all__ = ["AutoEngine"]


class Tray:
    """
    The `Tray` helper object that temporarily stores the engine components and
    attributes.

    Use `to_kwargs` to get the attachments as a kwargs dict
    """

    def to_kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        for var, value in self.__dict__.items():
            if var.startswith("__"):
                continue
            kwargs[_py_name(var)] = value
        return kwargs

    def __repr__(self) -> str:
        header = ["Tray()"]
        body = []
        for var, value in self.__dict__.items():
            if var.startswith("__"):
                continue
            body.append(" " * REPR_INDENT + f"{_py_name(var)}: {type(value)}")

        return "\n".join(header + body)

    __str__ = __repr__


class AutoEngine(Engine):
    """
    AutoEngine class.

    Example Usage:

    ```
    import torch
    import torch.nn as nn
    import torch.nn.functional as F


    tray = Tray()

    tray.train_loader = ...
    tray.eval_loader = ...

    tray.model = ...
    tray.optimizer = ...
    tray.scheduler = ...

    def train_step(tray, train_batch):

        image, target = train_batch
        pred = tray.model(image)
        loss = F.cross_entropy(pred, target)
        tray.optimizer.zero_grad()
        loss.backward()
        tray.optimizer.step()

        yield 'cross entropy loss', loss.item()

        acc = (pred.max(-1).indices == target).float().mean().item()

        yield 'train acc', acc


    def eval_step(tray, eval_batch):

        image, target = eval_batch
        with torch.no_grad():
            pred = tray.model(image)
        pred_ind = pred.max(-1)
        acc = (pred_ind == target).float().mean().item()

        yield 'test acc', acc

    train_buffers = AutoEngine.auto_buffers(train_step, ...)
    eval_buffers = AutoEngine.auto_buffers(eval_step, ...)
    tray.attach(**train_buffers)
    tray.attach(**eval_buffers)
    MyEngine = AutoEngine.build('ClassificationTrainer', train_step, eval_step)
    trainer = MyEngine(**tray.to_kwargs())

    ...
    ```
    """

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            if isinstance(v, EventHandler):
                self.attach_event(v)
            else:
                setattr(self, k, v)

    def per_batch(self, batch: Union[Tuple[Any], Dict[str, Any]], **kwargs: Any):
        if isinstance(self.current_stub, stub.Train):
            return self.train_step(batch, **kwargs)
        if isinstance(self.current_stub, stub.Evaluate):
            return self.eval_step(batch, **kwargs)
        if isinstance(self.current_stub, stub.Lambda):
            return getattr(self, self.current_stub.action)(batch, **kwargs)

        warnings.warn(f"Current stub type {type(self.current_stub)} is not recognized.")
        return

    @to_buffer()
    def train_step(self, batch: Any, **kwargs: Any) -> Generator:
        """
        if the train_step is a generator function, convert it to a method that
        pipes streaming outputs to buffer classes in `buffer_registry`

        Parameters
        ----------
        batch : Any
            Batch item

        Yields
        ------
        Generator
            Tuple[str, Union[float, Tensor]]
        """
        return self._train_step_generator(batch, **kwargs)

    @to_buffer()
    def eval_step(self, batch: Any, **kwargs: Any) -> Generator:
        """
        if the eval_step is a generator function, convert it to a method that
        pipes streaming outputs to buffer classes in `buffer_registry`

        Parameters
        ----------
        batch : Any
            Batch item

        Yields
        ------
        Generator
            Tuple[str, Union[float, Tensor]]
        """
        return self._eval_step_generator(batch, **kwargs)

    @classmethod
    def build(
        cls,
        engine_type_name: str,
        train_step: Optional[Callable] = None,
        eval_step: Optional[Callable] = None,
        **methods_to_attach: Callable,
    ) -> Type:
        cls_name = _py_name(engine_type_name)
        if train_step:
            if isgeneratorfunction(train_step):
                methods_to_attach["_train_step_generator"] = train_step
            elif isfunction(train_step):
                methods_to_attach["train_step"] = train_step
            else:
                raise ValueError(
                    "Method `train_step` should be a function"
                    " or generator function but got"
                    f" {type(train_step)}"
                )
        if eval_step:
            if isgeneratorfunction(eval_step):
                methods_to_attach["_eval_step_generator"] = eval_step
            elif isfunction(eval_step):
                methods_to_attach["eval_step"] = eval_step
            else:
                raise ValueError(
                    "Method `eval_step` should be a function"
                    " or generator function but got"
                    f" {type(eval_step)}"
                )

        return type(cls_name, (cls,), methods_to_attach)

    @staticmethod
    def auto_buffers(
        step_function: Generator, buffer_type: BufferBase, **buffer_kwargs
    ):
        names = [_py_name(n) for n in _find_output_names(step_function)]
        return {n: buffer_type(**buffer_kwargs) for n in names}
