import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import BufferBase

__all__ = [
    "MODEL",
    "OPTIMIZER",
    "SCHEDULER",
    "DATALOADER",
    "BUFFER",
    "COMPONENTS",
    "map_str_to_types",
    "map_types_to_str",
]

MODEL = (
    nn.Module,
    nn.ModuleList,
    nn.ModuleDict,
    torch.jit.ScriptModule,
)
OPTIMIZER = (optim.Optimizer,)
SCHEDULER = (optim.lr_scheduler._LRScheduler,)
DATALOADER = torch.utils.data.DataLoader
BUFFER = BufferBase
COMPONENTS = (
    MODEL,
    OPTIMIZER,
    SCHEDULER,
    DATALOADER,
    BUFFER,
)

map_str_to_types = {
    "model": MODEL,
    "optimizer": OPTIMIZER,
    "scheduler": SCHEDULER,
    "dataloader": DATALOADER,
    "buffer": BUFFER,
}

map_types_to_str = {v: k for k, v in map_str_to_types.items()}
