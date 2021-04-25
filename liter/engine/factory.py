import inspect
from .base import EngineBase
from .buffer import ScalarSmoother


def _find_outputs(generator):
    source = inspect.getsource(generator)
    names = []
    for line in source.splitlines():
        line = line.strip()
        if line.startswith("yield"):
            line = line.replace("yield", "")
            names.append(line.split(",")[0].strip()[1:-1])
    return names
