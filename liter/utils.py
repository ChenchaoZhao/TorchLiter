import copy
import importlib
from typing import *

__all__ = [
    "get_object_from_module",
    "instantiate_class",
    "build_instance_from_dict",
    "get_progress_bar",
]

VALUE_TYPES = (int, float, str, bool)
LIST_TYPES = (list, set, tuple)
DICT_TYPES = (dict,)


def get_object_from_module(module_path, object_name):
    m = importlib.import_module(module_path)
    c = getattr(m, object_name)
    return c


def instantiate_class(info: Dict):
    if "kwargs" in info:
        kwargs = info["kwargs"]
    else:
        kwargs = {}

    assert "module" in info, f"Key `module` not in {info}"
    assert "class" in info, f"Key `class` not in {info}"

    return get_object_from_module(info["module"], info["class"])(**kwargs)


def build_instance_from_dict(
    config: Dict, source_key: str, default_key: str = "default_params"
):
    """
    Params:
        config: Dict, config of the instance with keys
        ('instance_name_1', 'instance_name_2', default_key)

        source_key: str, name of the instance

        default_key: str, name of the key of default params in config
    """

    cfg = copy.deepcopy(config)

    default = cfg[default_key] if default_key in cfg else {}
    for k, v in default.items():
        assert v is not None, f"Default params should not be empty but {k} is empty."
    src = cfg[source_key]

    def replace(v, key=None):
        # replace None by default value
        # replace config_dict by python class

        if v is None:
            if key is None or key not in default:
                raise ValueError(
                    f"None type found with key {key} without replaceable default."
                )
            return replace(default[key])

        if isinstance(v, VALUE_TYPES):
            return v

        if isinstance(v, DICT_TYPES):
            for k, w in v.items():
                v[k] = replace(w, key=k)
            if "module" in v and "class" in v:
                return instantiate_class(v)
            return v

        if isinstance(v, LIST_TYPES):
            out = []
            for x in v:
                out.append(replace(x))
            return out

    if "kwargs" not in src:
        return instantiate_class(src)

    return replace(src)


def get_progress_bar(itr: int, tot: int, width: int = 25):

    done = int(itr / tot * width)
    todo = width - done
    progress_bar = "[" + "=" * (done) + " " * (todo) + "]"

    out = []
    out.append(f">> {itr+1}|{tot}")
    out.append(progress_bar)
    out.append(f"{int((itr+1)/tot * 100)}%")
    out.append(" " * 10)

    out = "  ".join(out)

    return out
