import abc
import collections
import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from .common import REPR_INDENT

MODEL = nn.Module
OPTIMIZER = optim.Optimizer
SCHEDULER = optim.lr_scheduler._LRScheduler
DATALOADER = torch.utils.data.DataLoader


class EngineBase(abc.ABC):

    _registry = (
        'model_registry',
        'optimizer_registry',
        'scheduler_registry',
        'dataloader_registry',
    )

    def __init__(self):
        for name in self._registry:
            super().__setattr__(name, {})
        self.reset_engine()

    def reset_engine(self):
        self.epoch = 0
        self.iteration = 0
        self.epoch_length = None
        self.stubs_in_queue = collections.deque()
        self.stubs_done = []
        self.current_stub = None

    def __setattr__(self, name, value):

        if name in self._registry:
            assert value == {}, f"{name} should be initialized with an empty dict()"
        else:
            if isinstance(value, MODEL):
                if name in self.model_registry:
                    warnings.warn(
                        f"Model with name {name} has already be registered. It will be overwritten."
                    )
                self.model_registry[name] = value
            elif isinstance(value, OPTIMIZER):
                if name in self.optimizer_registry:
                    warnings.warn(
                        f"Optimizer with name {name} has already be registered. It will be overwritten."
                    )
                self.optimizer_registry[name] = value
            elif isinstance(value, SCHEDULER):
                if name in self.scheduler_registry:
                    warnings.warn(
                        f"Scheduler with name {name} has already be registered. It will be overwritten."
                    )
                self.scheduler_registry[name] = value
            elif isinstance(value, DATALOADER):
                if name in self.dataloader_registry:
                    warnings.warn(
                        f"Dataloader with name {name} has already be registered. It will be overwritten."
                    )
                self.dataloader_registry[name] = value
            else:
                for r in self._registry:
                    assert name not in getattr(self, r), \
                    f"{name} has been registered in {r}"

            super().__setattr__(name, value)

    def __delattr__(self, name):

        value = getattr(self, name)

        if isinstance(value, MODEL):
            del self.model_registry[name]
        elif isinstance(value, OPTIMIZER):
            del self.optimizer_registry[name]
        elif isinstance(value, SCHEDULER):
            del self.scheduler_registry[name]
        elif isinstance(value, DATALOADER):
            del self.dataloader_registry[name]

        super().__delattr__(name)

    def state_dict(self):

        out = {}

        out['engine'] = {'epoch': self.epoch, 'iteration': self.iteration}
        out['model'] = {
            k: v.state_dict()
            for k, v in self.model_registry.items()
        }
        out['optimizer'] = {
            k: v.state_dict()
            for k, v in self.optimizer_registry.items()
        }
        out['scheduler'] = {
            k: v.state_dict()
            for k, v in self.scheduler_registry.items()
        }

        return out

    def load_state_dict(self, state_dict):

        if 'engine' in state_dict:
            epoch = int(state_dict['engine']['epoch'])
            iteration = int(state_dict['engine']['iteration'])
            if epoch < 0 or iteration < 0:
                self.reset_engine()
                warnings.warn(
                    "Invalid values encountered in engine state_dict.")
            else:
                self.epoch = epoch
                self.iteration = iteration
        else:
            self.reset_engine()

        for k, v in state_dict['model'].items():
            self.model_registry[k].load_state_dict(v)

        for k, v in state_dict['optimizer'].items():
            self.optimizer_registry[k].load_state_dict(v)

        for k, v in state_dict['scheduler'].items():
            self.scheduler_registry[k].load_state_dict(v)

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

    @abc.abstractmethod
    def per_batch(self, batch, **kwargs):
        pass

    @abc.abstractmethod
    def when_epoch_starts(self, **kwargs):
        pass

    @abc.abstractmethod
    def when_epoch_finishes(self, **kwargs):
        pass

    def queue(self, stubs):
        self.stubs_in_queue.extend(stubs)

    def execute(self, **kwargs):
        while len(self.stubs_in_queue) > 0:
            self.current_stub = self.stubs_in_queue.popleft()
            if self.current_stub.action == 'train':
                self.per_epoch(**kwargs)
            else:
                try:
                    action = getattr(self, self.current_stub.action)
                    action(**kwargs)
                except AttributeError as e:
                    warnings.warn(f"{e} Job aborted.")
            self.stubs_done.append(self.current_stub)

    def per_epoch(self, **kwargs):
        """Train model by one epoch"""
        dataloader = getattr(self, self.current_stub.dataloader)
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
            self.per_batch(batch, **kwargs)
            self.iteration += 1
            if self.iteration == self.epoch_length:
                # terminate such that total iterations equal epoch length
                # NOTE: so far assume sampling with replacement which is a good approximiation if batch is large
                # TODO: add sampling without replacement
                break

        self.epoch += 1
        self.when_epoch_finishes(**kwargs)

    def __call__(self, stubs=None, **kwargs):
        if stubs:
            self.queue(stubs)
        elif len(self.stubs_in_queue) == 0:
            raise ValueError("No job in queue. Stubs should not be empty")

        try:
            self.execute(**kwargs)
        except KeyboardInterrupt:
            self.current_stub.iteration = self.iteration
            self.stubs_in_queue.appendleft(self.current_stub)

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        out.append(' ' * REPR_INDENT + f"epoch: {self.epoch}")
        out.append(' ' * REPR_INDENT + f"iteration: {self.iteration}")

        # model
        out.append(' ' * REPR_INDENT + f"model: ")
        for k in self.model_registry:
            out.append(' ' * 2 * REPR_INDENT + f"{k}")

        # optimizer
        out.append(' ' * REPR_INDENT + f"optimizer: ")
        for k in self.optimizer_registry:
            out.append(' ' * 2 * REPR_INDENT + f"{k}")

        # scheduler
        out.append(' ' * REPR_INDENT + f"scheduler: ")
        for k in self.scheduler_registry:
            out.append(' ' * 2 * REPR_INDENT + f"{k}")

        # dataloader
        out.append(' ' * REPR_INDENT + f"dataloader: ")
        for k in self.dataloader_registry:
            out.append(' ' * 2 * REPR_INDENT + f"{k}")

        return '\n'.join(out)
