import warnings
from inspect import isfunction, isgeneratorfunction
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Type, Union

from .. import REPR_INDENT
from ..factory import FACTORY_PRODUCT_REGISTRY, FactoryRecord
from ..utils import _convert_str_to_py_object_name as _py_name
from .buffers import BufferBase, ExponentialMovingAverage, ScalarSummaryStatistics
from .events import Engine, EventHandler
from .utils import _find_output_names, to_buffer

__all__ = ["AutoEngine", "Cart"]


class Cart:
    """
    The `Cart` helper object that temporarily stores the engine components and
    attributes.

    - Use `kwargs` to get the attachments as a kwargs dict
    - Use `attach(**kwargs)` to attach attributes in bulk
    """

    attachment_records: Dict[str, FactoryRecord]

    def __init__(self, *args: Any, **kwargs: Any):
        self.attachment_records = {}
        self.attach(*args, **kwargs)

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = {}
        for var, value in self.__dict__.items():
            if var.startswith("__") or var == "attachment_records":
                continue
            kwargs[_py_name(var)] = value
        return kwargs

    def attach(self, *args, **kwargs):
        if args:
            raise RuntimeError("Only keyword args allowed.")
        for var, value in kwargs.items():
            setattr(self, var, value)

    def __setattr__(self, name: str, obj: Any) -> None:
        _id = id(obj)
        if _id in FACTORY_PRODUCT_REGISTRY:
            record = FACTORY_PRODUCT_REGISTRY[_id]
            self.attachment_records[name] = record
        super().__setattr__(name, obj)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        if name in self.attachment_records:
            self.attachment_records.pop(name)

    def parse_buffers(
        self,
        step_function: Generator,
        mode: Optional[str] = None,
        buffer_type: Optional[BufferBase] = None,
        **kwargs,
    ):
        if not mode:
            assert buffer_type, "If mode is None, buffer_type must be specified"
        elif mode == "train":
            if not buffer_type:
                buffer_type = ExponentialMovingAverage
        elif mode == "eval":
            if not buffer_type:
                buffer_type = ScalarSummaryStatistics
        buffers = AutoEngine.auto_buffers(step_function, buffer_type, **kwargs)
        self.attach(**buffers)

    def __len__(self) -> int:
        return len(self.kwargs)

    def __repr__(self) -> str:
        header = [self.__class__.__name__]
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


     cart = torchliter.Cart()
     cart.model = nn.Linear(1, 3)
     cart.train_loader = torch.utils.data.DataLoader(
         [i for i in range(100)], batch_size=5
     )
     cart.eval_loader = torch.utils.data.DataLoader(
         [i for i in range(100)], batch_size=5
     )
     cart.optimizer = torch.optim.AdamW(
         cart.model.parameters(), lr=1e-3, weight_decay=1e-5
     )

     def train_step(_, batch, **kwargs):
         image, target = batch
         logits = _.model(image)
         loss = F.cross_entropy(logits, target)
         _.optimizer.zero_grad()
         loss.backward()
         _.optimizer.step()

         yield "cross entropy loss", loss.item()

         acc = (logits.max(-1).indices == target).float().mean()

         yield "train acc", acc.item()

     def eval_step(_, batch, **kwargs):
         image, target = batch
         with torch.no_grad():
             logits = _.model(image)
         acc = (logits.max(-1).indices == target).float().mean()
         yield "eval acc", acc.item()

     def hello(_):
         print("hello")

     train_buffers = torchliter.engine.AutoEngine.auto_buffers(
         train_step, torchliter.buffers.ExponentialMovingAverage
     )
     eval_buffers = torchliter.engine.AutoEngine.auto_buffers(
         eval_step, torchliter.buffers.ScalarSummaryStatistics
     )
     TestEngineClass = torchliter.engine.AutoEngine.build(
         "TestEngine", train_step, eval_step, print_hello=hello
     )
     test_engine = TestEngineClass(**{**cart.kwargs, **train_buffers, **eval_buffers})

    ```
    """

    def __init__(self, *events, **kwargs):
        super().__init__()
        self.attach(*events, **kwargs)

    def attach(self, *events, **kwargs):
        for event in events:
            if not isinstance(event, EventHandler):
                raise RuntimeError("Positional args can only be `EventHandler`s.")
            self.attach_event(event)
        for k, v in kwargs.items():
            if isinstance(v, EventHandler):
                self.attach_event(v)
            else:
                setattr(self, k, v)

    def per_batch(self, batch: Union[Tuple[Any], Dict[str, Any]], **kwargs: Any):
        if self.is_train_stub:
            # set to train mode automatically
            self.train()
            return self.train_step(batch, **kwargs)
        if self.is_eval_stub:
            # set to eval model automatically
            self.eval()
            return self.eval_step(batch, **kwargs)
        if self.is_lambda_stub:
            return getattr(self, self.current_stub.action)(batch, **kwargs)

        warnings.warn(f"Current stub type {type(self.current_stub)} is not recognized.")

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
