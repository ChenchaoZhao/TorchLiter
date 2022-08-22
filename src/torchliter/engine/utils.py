import inspect
from typing import Any, Generator, List, Tuple

__all__ = ["_find_output_names"]


def _find_output_names(func: Generator[Tuple[str, Any]]) -> List[str]:
    """
    Returns the variable names yielded from the generator.

    Parameters
    ----------
    func : Generator[Tuple[str, Any]]
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
            names.append(line.split(",")[0].strip()[1:-1])
    return names
