<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.engine.factory`






---

## <kbd>class</kbd> `Automated`
Automated Engine Given a forward generator function, `from_forward` will return an Automated engine class. 

For example ================== 

import torch import torch.nn as nn import torch.nn.functional as F 

@Automated.config(smooth_window=100) def classification(engine, batch):  # the first arg must be a place holder for engine class 

 engine.train()  x, y = batch  lgs = engine.model(x)  loss = F.cross_entropy(lgs, y) 

 yield "loss", loss.item()  # metrics will be registered as buffers 

 acc = (lgs.max(-1).indices == y).float().mean() 

 yield "acc", acc.item() 

# or alternatively # `from_forward` will be deprecated classification = Automated.from_forward(classification) 

# attach other components such as model, optimizer, dataloader, etc. eng.attach(model=nn.Linear(2, 2)) ... 

### <kbd>method</kbd> `Automated.__init__`

```python
__init__(
    core_function:Callable,
    alpha:float=0.01,
    smooth_window:int=50,
    buffer_type:liter.engine.buffer.BufferBase=<class 'liter.engine.buffer.ExponentialMovingAverage'>,
    **kwargs
)
```






---

#### <kbd>property</kbd> Automated.absolute_iterations





---

#### <kbd>property</kbd> Automated.fractional_epoch





---

#### <kbd>property</kbd> Automated.fractional_iteration





---

#### <kbd>property</kbd> Automated.training







---

### <kbd>method</kbd> `Automated.attach`

```python
attach(**kwargs)
```





---

### <kbd>classmethod</kbd> `Automated.config`

```python
config(**kwargs)
```

Used as decorator for core function allowing user to attach additional init keyword args Examples. 

@Automated def core_func(ng, batch):  ... 



@Automated.config(smooth_window=100) def core_func(ng, batch):  ... 

---

### <kbd>method</kbd> `Automated.core`

```python
core(batch, **kwargs)
```





---

### <kbd>classmethod</kbd> `Automated.from_forward`

```python
from_forward(func, smooth_window=50, **kwargs)
```

This method is deprecated. 

Use init as decorator or cls.config(...) as decorator 

---

### <kbd>method</kbd> `Automated.per_batch`

```python
per_batch(batch, **kwargs)
```






