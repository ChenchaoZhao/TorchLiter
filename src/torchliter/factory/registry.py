import functools
import inspect
import warnings
from typing import Any, Callable, Dict, NamedTuple

__all__ = [
    "FactoryRecord",
    "FACTORY_FUNCTION_REGISTRY",
    "FACTORY_PRODUCT_REGISTRY",
    "register_factory",
]


class FactoryRecord(NamedTuple):
    """
    Factory record object.

    Parameters
    ----------
    factory_function_name: str
    input_parameters: Dict[str, Any]
    """

    factory_function_name: str
    input_parameters: Dict[str, Any]


FACTORY_FUNCTION_REGISTRY: Dict[str, inspect.Signature] = {}
FACTORY_PRODUCT_REGISTRY: Dict[int, FactoryRecord] = {}


def register_factory(factory_function: Callable) -> Callable:
    assert inspect.isfunction(factory_function)
    _factory_name = ".".join([factory_function.__module__, factory_function.__name__])
    _signature = inspect.signature(factory_function)

    FACTORY_FUNCTION_REGISTRY[_factory_name] = _signature

    @functools.wraps(factory_function)
    def auto_logging_factory_function(*args, **kwargs):

        _kwargs = {}
        _args = list(reversed(args))

        for var, par in _signature.parameters.items():
            if _args:
                arg_val = _args.pop()
                _kwargs[var] = arg_val
            else:
                if var in kwargs:
                    _kwargs[var] = kwargs.pop(var)
                else:  # fill in default value
                    _kwargs[var] = par.default
        assert not _args, f"More args than expected {_args[::-1]}."
        assert not kwargs, f"More kwargs than expected {kwargs}."

        product = factory_function(**_kwargs)

        if isinstance(product, (list, tuple, set, dict)):
            warnings.warn(
                f"Return of `{factory_function.__name__}` is a container object. "
                "If the container object contains distinct components, they should "
                "be created separately. Otherwise they cannot be tracked correctly."
            )

        memory_address = id(product)
        FACTORY_PRODUCT_REGISTRY[memory_address] = FactoryRecord(_factory_name, _kwargs)

        return product

    return auto_logging_factory_function
