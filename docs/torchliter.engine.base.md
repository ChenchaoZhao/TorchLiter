<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.engine.base`




**Global Variables**
---------------
- **REPR_INDENT**
- **COMPONENTS**
- **map_str_to_types**
- **map_types_to_str**


---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L13"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EngineBase`
Base class of Engine classes. 

Attributes 
---------- epoch : int  Current epoch iteration : int  Current interation fractional_epoch: float  fractional_epoch = epoch + iteration/epoch_length fractional_iteration : float  fractional_iteration = iteration/epoch_length epoch_length : Optional[int]  Total number of iterations in an epoch absolute_iterations : int  absolute_iterations = epoch_length * epoch + iteration _registry : Tuple[Dict[str, Any]]  Registry of engine components. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.__init__`

```python
__init__()
```






---

#### <kbd>property</kbd> EngineBase.absolute_iterations





---

#### <kbd>property</kbd> EngineBase.fractional_epoch





---

#### <kbd>property</kbd> EngineBase.fractional_iteration





---

#### <kbd>property</kbd> EngineBase.is_eval_stub





---

#### <kbd>property</kbd> EngineBase.is_lambda_stub





---

#### <kbd>property</kbd> EngineBase.is_train_stub





---

#### <kbd>property</kbd> EngineBase.training







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.after_iteration`

```python
after_iteration(**kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.before_iteration`

```python
before_iteration(**kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L189"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.eval`

```python
eval()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L312"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.execute`

```python
execute(**kwargs: Any) → None
```

Executes stubs in queue. 

Parameters 
---------- **kwargs : Any 

Returns 
------- None 

Raises 
------ AttributeError  No action attached to current stub. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L150"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.load_state_dict`

```python
load_state_dict(state_dict: Dict[str, Dict[str, Any]]) → None
```

Load state into engine. 

Parameters 
---------- state_dict : Dict[str, Dict[str, Any]]  Engine state dict. 

Returns 
------- None 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.per_batch`

```python
per_batch(batch: Union[Tuple[Any], Dict[str, Any]], **kwargs: Any)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.per_epoch`

```python
per_epoch(**kwargs)
```

Train, eval model or performe a lambda op by one epoch. 

The stub must have `dataloader`. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L293"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.queue`

```python
queue(stubs: List[torchliter.stub.StubBase]) → None
```

Adds stubs to queue. 

Parameters 
---------- stubs : List[StubBase]  A list of `stubs`. 

Returns 
------- None 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L51"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.reset_engine`

```python
reset_engine() → None
```

Reset engine state. 

epoch -> 0 iteration -> 0 stubs -> [] 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L308"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.reset_queue`

```python
reset_queue() → None
```

Resets stubs queue to empty. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L126"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.state_dict`

```python
state_dict() → Dict[str, Dict[str, Any]]
```

Generate current state of the engine as `Dict`. 

Returns 
------- Dict[str, Dict[str, Any]]  Dict of component state dicts. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.train`

```python
train()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.when_epoch_finishes`

```python
when_epoch_finishes(**kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/base.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EngineBase.when_epoch_starts`

```python
when_epoch_starts(**kwargs)
```






