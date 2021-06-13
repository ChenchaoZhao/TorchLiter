import collections
import warnings
from typing import *

from .. import REPR_INDENT
from ..exception import BreakIteration, ContinueIteration
from ..stub import StubBase
from .component_types import *

__all__ = ["EngineBase"]


class EngineBase:

    epoch: int
    iteration: int
    fractional_epoch: float
    fractional_iteration: float
    epoch_length: Optional[int]
    absolute_iterations: int

    _registry: Tuple[Dict[str, Any]] = tuple(
        [f"{c}_registry" for c in map_str_to_types]
    )

    def __init__(self):
        for name in self._registry:
            object.__setattr__(self, name, {})
        self.reset_engine()

    def reset_engine(self):
        self.epoch = 0
        self.iteration = 0
        self.epoch_length = None
        self.stubs_in_queue = collections.deque()
        self.stubs_done = []
        self.current_stub = None

    def __setattr__(self, name: str, value: Any):
        """
        set attribute operator.

        - if name exists
          - if current attr is engine component:
          raise current attr is a component please del it first
          - if current attr is a registry: raise registry is immutable
          - else: treat it as a new attr
        - if name is new
          - if value is a component: register in registry
          - else: pass
          - object set attr
        """

        if hasattr(self, name):

            if isinstance(getattr(self, name), COMPONENTS):
                raise AttributeError(
                    f"Attribute `{name}` is an engine component."
                    " You need to delete it first before assigning a new value."
                )

            if name == "_registry":
                raise AttributeError("`_registry` is a protected attribute.")

            if name in self._registry:
                raise AttributeError(f"`{name}` is immutable.")

        if isinstance(value, COMPONENTS):

            for TYPE in COMPONENTS:
                if isinstance(value, TYPE):
                    typestr = map_types_to_str[TYPE]
                    break
            registry = getattr(self, f"{typestr}_registry")
            assert (
                name not in registry
            ), f"The `{name}` is already in `{typestr}_registry`. "
            "Components should be registered and deregistered "
            "through setting and deleting attribues."
            registry[name] = value

        object.__setattr__(self, name, value)

    def __delattr__(self, name: str):
        """delete attribute operator if attr is a component, also deregister the
        component in the corresponding registry."""

        value = getattr(self, name)

        if isinstance(value, COMPONENTS):

            for TYPE in COMPONENTS:
                if isinstance(value, TYPE):
                    typestr = map_types_to_str[TYPE]
                    break
            registry = getattr(self, f"{typestr}_registry")
            del registry[name]

        object.__delattr__(self, name)

    def state_dict(self) -> Dict[str, Dict[str, Any]]:
        out = {}

        out["engine"] = {"epoch": self.epoch, "iteration": self.iteration}

        for rname in self._registry:
            registry = getattr(self, rname)
            cname = rname.split("_")[0]  # e.g. `model`
            if (
                cname == "dataloader"
            ):  # current dataloader class doesn't have state dict
                continue
            out[cname] = {k: v.state_dict() for k, v in registry.items()}

        return out

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]):

        if "engine" not in state_dict:
            self.reset_engine()

        for cname, cstate in state_dict.items():

            if cname == "engine":
                epoch = int(state_dict["engine"]["epoch"])
                iteration = int(state_dict["engine"]["iteration"])
                if epoch < 0 or iteration < 0:
                    self.reset_engine()
                    warnings.warn("Invalid values encountered in engine state_dict.")
                else:
                    self.epoch = epoch
                    self.iteration = iteration
                continue

            # otherwise
            registry = getattr(self, f"{cname}_registry")
            for k, v in cstate.items():
                registry[k].load_state_dict(v)

    def train(self):
        for k, v in self.model_registry.items():
            v.train()

    def eval(self):
        for k, v in self.model_registry.items():
            v.eval()

    @property
    def training(self):
        out = {}
        for k, v in self.model_registry.items():
            out[k] = v.training
        return out

    def per_batch(self, batch: Union[Tuple[Any], Dict[str, Any]], **kwargs: Any):
        raise NotImplementedError("Method `per_batch` is not implemented")

    def when_epoch_starts(self, **kwargs):
        pass

    def when_epoch_finishes(self, **kwargs):
        pass

    def between_iterations(self, **kwargs):
        pass

    @property
    def fractional_epoch(self) -> float:
        return self.epoch + self.iteration / self.epoch_length

    @property
    def fractional_iteration(self) -> float:
        return self.iteration / self.epoch_length

    @property
    def absolute_iterations(self) -> int:
        return self.epoch * self.epoch_length + self.iteration

    def per_epoch(self, **kwargs):
        """Train model by one epoch."""

        if self.current_stub.dataloader in self.dataloader_registry:
            dataloader = getattr(self, self.current_stub.dataloader)
        else:
            raise AttributeError(
                f"`{self.current_stub.dataloader}` is not in dataloader registry."
            )

        self.epoch_length = len(dataloader)
        self.when_epoch_starts(**kwargs)
        try:
            # if current stub was paused, then pick up from there
            it = self.current_stub.iteration
            it = int(it)
            self.iteration = it if it > 0 else 0
        except AttributeError as e:
            # if stub does not have iteration
            warnings.warn(f"{e} Iteration reset to zero.")
            self.iteration = 0

        for batch in dataloader:
            try:
                self.per_batch(batch, **kwargs)
                self.iteration += 1
                self.between_iterations(**kwargs)
                if self.iteration == self.epoch_length:
                    # terminate such that total iterations equal epoch length
                    # NOTE: so far assume sampling with replacement
                    # which is a good approximiation if batch is large
                    # TODO: add sampling without replacement
                    raise BreakIteration(False)
            except ContinueIteration:
                continue
            except BreakIteration as e:
                if e.shutdown_engine:
                    raise BreakIteration
                break
            except Exception as e:
                raise e

        self.epoch += 1
        self.when_epoch_finishes(**kwargs)

    def queue(self, stubs: List[StubBase]):
        self.stubs_in_queue.extend(stubs)

    def execute(self, **kwargs: Any):
        while len(self.stubs_in_queue) > 0:
            self.current_stub = self.stubs_in_queue.popleft()
            if self.current_stub.action == "train":
                try:
                    self.per_epoch(**kwargs)
                except BreakIteration:
                    break
                except ContinueIteration:
                    continue
            else:
                try:
                    action = getattr(self, self.current_stub.action)
                    action(**kwargs)
                except AttributeError as e:
                    warnings.warn(f"{e} Job aborted.")
                    continue
            self.stubs_done.append(self.current_stub)

    def __call__(self, stubs: Optional[List[StubBase]] = None, **kwargs: Any):
        if stubs:
            self.queue(stubs)
        elif len(self.stubs_in_queue) == 0:
            raise ValueError("No job in queue. Stubs should not be empty")

        try:
            self.execute(**kwargs)
        except KeyboardInterrupt:
            self.current_stub.iteration = self.iteration
            self.stubs_in_queue.appendleft(self.current_stub)

    def __repr__(self) -> str:
        out = []
        out.append(self.__class__.__name__)
        out.append(" " * REPR_INDENT + f"epoch: {self.epoch}")
        out.append(" " * REPR_INDENT + f"iteration: {self.iteration}")

        for registry in self._registry:
            out.append(" " * REPR_INDENT + f"{registry}: ")
            for k in getattr(self, registry):
                out.append(" " * 2 * REPR_INDENT + f"{k}")

        return "\n".join(out)
