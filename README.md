# TorchLiter
[![test](https://github.com/ChenchaoZhao/TorchLiter/actions/workflows/lint-test.yaml/badge.svg)](https://github.com/ChenchaoZhao/TorchLiter/actions/workflows/lint-test.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/torchliter.svg)](https://badge.fury.io/py/torchliter)

A freely customizable and truly lightweight training tool for any pytorch projects
## Install
```
pip install torchliter
```
## Example Usage:

```python
import torchliter as lux
import torch
import torch.nn as nn
import torch.nn.functional as F


cart = lux.Cart()
cart.model = nn.Linear(1, 3)
cart.train_loader = torch.utils.data.DataLoader(
    [i for i in range(100)], batch_size=5
)
cart.eval_loader = torch.utils.data.DataLoader(
    [i for i in range(100)], batch_size=5
)
cart.optimizer = torch.optim.AdamW(
    cart.model.parameters(), lr=1e-3, weight_decay=1e-5
)

def train_step(_, batch, **kwargs):
    image, target = batch
    logits = _.model(image)
    loss = F.cross_entropy(logits, target)
    _.optimizer.zero_grad()
    loss.backward()
    _.optimizer.step()

    yield "cross entropy loss", loss.item()

    acc = (logits.max(-1).indices == target).float().mean()

    yield "train acc", acc.item()

def eval_step(_, batch, **kwargs):
    image, target = batch
    with torch.no_grad():
        logits = _.model(image)
    acc = (logits.max(-1).indices == target).float().mean()
    yield "eval acc", acc.item()

def hello(_):
    print("hello")

train_buffers = lux.engine.AutoEngine.auto_buffers(
    train_step, lux.buffers.ExponentialMovingAverage
)
eval_buffers = lux.engine.AutoEngine.auto_buffers(
    eval_step, lux.buffers.ScalarSummaryStatistics
)
TestEngineClass = lux.engine.AutoEngine.build(
    "TestEngine", train_step, eval_step, print_hello=hello
)
test_engine = TestEngineClass(**{**cart.kwargs, **train_buffers, **eval_buffers})

```
