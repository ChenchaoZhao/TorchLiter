import hashlib
from typing import Any

__all__ = ["get_md5_hash"]


def get_md5_hash(obj: Any) -> str:
    return hashlib.md5(_repr_string(obj).encode()).hexdigest()


def _repr_string(obj: Any) -> str:

    if isinstance(obj, (list, tuple, set)):
        return str([_repr_string(item) for item in obj])
    if isinstance(obj, dict):
        return str({_repr_string(key): _repr_string(val) for key, val in obj.items()})

    if hasattr(obj, "__dict__"):
        return f"object: {repr(obj)} with attrs {repr(obj.__dict__)}"
    else:
        return f"object: {repr(obj)}"
