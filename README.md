# TorchLiter
[![test](https://github.com/ChenchaoZhao/TorchLiter/actions/workflows/lint-test.yaml/badge.svg)](https://github.com/ChenchaoZhao/TorchLiter/actions/workflows/lint-test.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/torchliter.svg)](https://badge.fury.io/py/torchliter)

A freely customizable and truly lightweight training tool for any pytorch projects
## Install
```
pip install torchliter
```
## Documentation
TorchLiter full documentation is [here](https://chenchaozhao.github.io/TorchLiter/) where the most important class, `Automated` engine class is described  [here](https://chenchaozhao.github.io/TorchLiter/liter/engine/factory.html).

Example usage:

```python
import liter
import torch
import torch.nn as nn
import torch.nn.functional as F

@liter.engine.Automated.config(smooth_window=100)
def classification(engine, batch):
    # the first arg must be a place holder for engine class

    engine.train()
    x, y = batch
    lgs = engine.model(x)
    loss = F.cross_entropy(lgs, y)

    yield "loss", loss.item()
    # metrics will be registered as buffers

    acc = (lgs.max(-1).indices == y).float().mean()

    yield "acc", acc.item()
```
