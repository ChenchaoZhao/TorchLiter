import inspect
from functools import wraps
from typing import Any, Callable, Generator, List, Tuple

from ..utils import _convert_str_to_py_object_name as _py_name

__all__ = ["to_buffer", "_find_output_names"]


def to_buffer(buffer_registry_name="buffer_registry") -> Callable:
    """
    Returns a decorator that push the updates to corresponding buffers.

    For example,

    ```
    @to_buffer('some-buffer-registry'):
    def some_step_method(self, *args):
        ...
        yield 'var1', var1
        ...
        yield 'var2', var2
    ```
    where `var1` and `var2` are buffer names in `some-buffer-registry`.


    Parameters
    ----------
    buffer_registry_name : str, optional
        name of buffer registry, by default "buffer_registry"

    Returns
    -------
    Callable
        A decorator that turns a generator to a method the automatically
        pushes updates to buffers
    """
    # name should be an attribute of the owner class

    def decorator(func: Generator[Tuple[str, Any], None, None]):
        # func: class method that yields tuple of (key: str, val)
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            buffer_dict = getattr(self, buffer_registry_name)
            for key, val in func(self, *args, **kwargs):
                key = _py_name(key)
                if key in buffer_dict:
                    buffer_dict[key](val)  # pushing update by `__call__`

        return wrapper

    return decorator


def _find_output_names(func: Generator[Tuple[str, Any], None, None]) -> List[str]:
    """
    Returns the variable names yielded from the generator.

    Parameters
    ----------
    func : Generator[Tuple[str, Any], None, None]
        A generator that returns tuple of (variable name, variable value)

    Returns
    -------
    List[str]
        Variable names
    """

    assert inspect.isgeneratorfunction(func), (
        "The forward function must be a generator function "
        "with first arg being engine class placeholder."
    )

    source = inspect.getsource(func)
    names = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("yield"):
            line = line.replace("yield", "")
            names.append(line.rsplit(",", 1)[0].strip()[1:-1])
            # use rsplit because var names can be e.g. "a,b,c"
    return names
