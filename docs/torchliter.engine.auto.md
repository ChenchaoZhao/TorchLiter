<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.engine.auto`




**Global Variables**
---------------
- **REPR_INDENT**
- **FACTORY_PRODUCT_REGISTRY**


---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Cart`
The `Cart` helper object that temporarily stores the engine components and attributes. 


- Use `kwargs` to get the attachments as a kwargs dict 
- Use `attach(**kwargs)` to attach attributes in bulk 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cart.__init__`

```python
__init__(*args: Any, **kwargs: Any)
```






---

#### <kbd>property</kbd> Cart.kwargs







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cart.attach`

```python
attach(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Cart.parse_buffers`

```python
parse_buffers(
    step_function: Generator,
    mode: Optional[str] = None,
    buffer_type: Optional[torchliter.engine.buffers.BufferBase] = None,
    **kwargs
)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `AutoEngine`
AutoEngine class. 

Example Usage: 

```
 import torch
 import torch.nn as nn
 import torch.nn.functional as F


 cart = torchliter.Cart()
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

 train_buffers = torchliter.engine.AutoEngine.auto_buffers(
      train_step, torchliter.buffers.ExponentialMovingAverage
 )
 eval_buffers = torchliter.engine.AutoEngine.auto_buffers(
      eval_step, torchliter.buffers.ScalarSummaryStatistics
 )
 TestEngineClass = torchliter.engine.AutoEngine.build(
      "TestEngine", train_step, eval_step, print_hello=hello
 )
 test_engine = TestEngineClass(**{**cart.kwargs, **train_buffers, **eval_buffers})

``` 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L153"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.__init__`

```python
__init__(*events, **kwargs)
```






---

#### <kbd>property</kbd> AutoEngine.absolute_iterations





---

#### <kbd>property</kbd> AutoEngine.fractional_epoch





---

#### <kbd>property</kbd> AutoEngine.fractional_iteration





---

#### <kbd>property</kbd> AutoEngine.is_eval_stub





---

#### <kbd>property</kbd> AutoEngine.is_lambda_stub





---

#### <kbd>property</kbd> AutoEngine.is_train_stub





---

#### <kbd>property</kbd> AutoEngine.training







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.attach`

```python
attach(*events, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.auto_buffers`

```python
auto_buffers(
    step_function: Generator,
    buffer_type: torchliter.engine.buffers.BufferBase,
    **buffer_kwargs
)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L218"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `AutoEngine.build`

```python
build(
    engine_type_name: str,
    train_step: Optional[Callable] = None,
    eval_step: Optional[Callable] = None,
    **methods_to_attach: Callable
) → Type
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/utils.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.eval_step`

```python
eval_step(batch: Any, **kwargs: Any) → Generator
```

if the eval_step is a generator function, convert it to a method that pipes streaming outputs to buffer classes in `buffer_registry` 

Parameters 
---------- batch : Any  Batch item 

Yields 
------ Generator  Tuple[str, Union[float, Tensor]] 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/auto.py#L168"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.per_batch`

```python
per_batch(batch: Union[Tuple[Any], Dict[str, Any]], **kwargs: Any)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/utils.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `AutoEngine.train_step`

```python
train_step(batch: Any, **kwargs: Any) → Generator
```

if the train_step is a generator function, convert it to a method that pipes streaming outputs to buffer classes in `buffer_registry` 

Parameters 
---------- batch : Any  Batch item 

Yields 
------ Generator  Tuple[str, Union[float, Tensor]] 


