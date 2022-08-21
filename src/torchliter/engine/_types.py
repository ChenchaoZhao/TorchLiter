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
    "GRADSCALER",
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
SCHEDULER = (optim.lr_scheduler._LRScheduler, optim.lr_scheduler.ReduceLROnPlateau)
GRADSCALER = (torch.cuda.amp.GradScaler,)
DATALOADER = torch.utils.data.DataLoader
BUFFER = BufferBase
COMPONENTS = (MODEL, OPTIMIZER, SCHEDULER, DATALOADER, BUFFER, GRADSCALER)

map_str_to_types = {
    "model": MODEL,
    "optimizer": OPTIMIZER,
    "scheduler": SCHEDULER,
    "dataloader": DATALOADER,
    "buffer": BUFFER,
    "gradscaler": GRADSCALER,
}

map_types_to_str = {v: k for k, v in map_str_to_types.items()}
