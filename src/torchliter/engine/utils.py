import inspect
from typing import Generator, List

__all__ = ["_find_output_names"]


def _find_output_names(func: Generator) -> List[str]:

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
