import warnings
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

from .. import stub
from ..utils import _convert_str_to_py_object_name as _py_name
from .events import Engine, EventHandler

__all__ = ["AutoEngine"]


class AutoEngine(Engine):
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

    @classmethod
    def _create_new_engine_type(
        cls,
        engine_type_name: str,
        train_step: Optional[Callable] = None,
        eval_step: Optional[Callable] = None,
        **methods_to_attach: Callable,
    ) -> Type:
        cls_name = _py_name(engine_type_name)
        if train_step:
            methods_to_attach["train_step"] = train_step
        if eval_step:
            methods_to_attach["eval_step"] = eval_step
        return type(cls_name, (cls,), methods_to_attach)
