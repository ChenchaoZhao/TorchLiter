from typing import Callable, Optional, Type

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

    @classmethod
    def _new_engine_type(
        cls,
        engine_type_name: str,
        train_step: Optional[Callable] = None,
        eval_step: Optional[Callable] = None,
        **methods_to_attach: Callable
    ) -> Type:
        cls_name = _py_name(engine_type_name)
        if train_step:
            methods_to_attach["train_step"] = train_step
        if eval_step:
            methods_to_attach["eval_step"] = eval_step
        return type(cls_name, (cls,), methods_to_attach)
