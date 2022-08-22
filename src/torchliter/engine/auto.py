from typing import Callable, Type

from ..utils import _convert_str_to_py_object_name as _py_name
from .events import Engine

__all__ = ["AutoEngine"]


class EngineMeta(Type):
    def __new__(cls, name, bases, attrs):
        name = _py_name(name)
        attrs = {_py_name(k): v for k, v in attrs.items()}
        return super().__new__(cls, name, bases, attrs)


class AutoEngine(Engine):
    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def _new_engine_type(
        cls, engine_type_name: str, **methods_to_attach: Callable
    ) -> Type:
        cls_name = _py_name(engine_type_name)
        return type(cls_name, (cls,), methods_to_attach)
