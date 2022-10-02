<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.engine.events`




**Global Variables**
---------------
- **REPR_INDENT**


---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EventCategory`
An enumeration. 





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EventHandler`
Base Class for Event Handlers. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EventHandler.__init__`

```python
__init__(
    action_function: Optional[Callable[[torchliter.engine.base.EngineBase], NoneType]] = None,
    trigger_function: Optional[Callable[[torchliter.engine.base.EngineBase], bool]] = None,
    **kwargs
)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EventHandler.action`

```python
action(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `EventHandler.config`

```python
config(*args, **kwargs) → Type
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EventHandler.trigger`

```python
trigger(engine: torchliter.engine.base.EngineBase) → bool
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PreEpochHandler`
Hanldes events when a new epoch starts. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreEpochHandler.__init__`

```python
__init__(
    action_function: Callable[[torchliter.engine.base.EngineBase], NoneType],
    trigger_function: Optional[Callable[[torchliter.engine.base.EngineBase], bool]] = None,
    every: int = 1,
    train_stub: bool = True,
    eval_stub: bool = True,
    lambda_stub: bool = False
)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreEpochHandler.action`

```python
action(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `PreEpochHandler.config`

```python
config(*args, **kwargs) → Type
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreEpochHandler.trigger`

```python
trigger(engine: torchliter.engine.base.EngineBase) → bool
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PostEpochHandler`
Hanldes events when a new epoch finishes. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L257"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostEpochHandler.__init__`

```python
__init__(
    action_function: Callable[[torchliter.engine.base.EngineBase], NoneType],
    trigger_function: Optional[Callable[[torchliter.engine.base.EngineBase], bool]] = None,
    every: int = 1,
    train_stub: bool = True,
    eval_stub: bool = True,
    lambda_stub: bool = False
)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostEpochHandler.action`

```python
action(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `PostEpochHandler.config`

```python
config(*args, **kwargs) → Type
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostEpochHandler.trigger`

```python
trigger(engine: torchliter.engine.base.EngineBase) → bool
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PreIterationHandler`
Handles events before each iteration. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreIterationHandler.__init__`

```python
__init__(
    action_function: Callable[[torchliter.engine.base.EngineBase], NoneType],
    trigger_function: Optional[Callable[[torchliter.engine.base.EngineBase], bool]] = None,
    every: int = 1,
    train_stub: bool = True,
    eval_stub: bool = True,
    lambda_stub: bool = False
)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreIterationHandler.action`

```python
action(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `PreIterationHandler.config`

```python
config(*args, **kwargs) → Type
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PreIterationHandler.trigger`

```python
trigger(engine: torchliter.engine.base.EngineBase) → bool
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `PostIterationHandler`
Handles events after each iteration. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostIterationHandler.__init__`

```python
__init__(
    action_function: Callable[[torchliter.engine.base.EngineBase], NoneType],
    trigger_function: Optional[Callable[[torchliter.engine.base.EngineBase], bool]] = None,
    every: int = 1,
    train_stub: bool = True,
    eval_stub: bool = True,
    lambda_stub: bool = False
)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostIterationHandler.action`

```python
action(*args, **kwargs)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `PostIterationHandler.config`

```python
config(*args, **kwargs) → Type
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L41"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `PostIterationHandler.trigger`

```python
trigger(engine: torchliter.engine.base.EngineBase) → bool
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Engine`
Engine with Event Handler plugin. 

Attributes 
---------- _event_handlers : Dict[EventCategory, List[EventHandler]]  The registry of event handlers. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L342"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.__init__`

```python
__init__()
```






---

#### <kbd>property</kbd> Engine.absolute_iterations





---

#### <kbd>property</kbd> Engine.fractional_epoch





---

#### <kbd>property</kbd> Engine.fractional_iteration





---

#### <kbd>property</kbd> Engine.is_eval_stub





---

#### <kbd>property</kbd> Engine.is_lambda_stub





---

#### <kbd>property</kbd> Engine.is_train_stub





---

#### <kbd>property</kbd> Engine.training







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L398"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.after_iteration`

```python
after_iteration()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L351"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.attach_event`

```python
attach_event(handler: torchliter.engine.events.EventHandler)
```

Attach an event handler. 

Parameters 
---------- handler : EventHandler  An event handler to be attached. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L394"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.before_iteration`

```python
before_iteration()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.list_events`

```python
list_events(
    event_category: Optional[str] = None
) → Tuple[torchliter.engine.events.EventHandler]
```

List events based on category. 

Parameters 
---------- event_category : Optional[str]  List event handlers in `event_category` (the default is None).  If not provided, list all handlers. 

Returns 
------- Tuple[EventHandler]  Tuple of handlers. 

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L390"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.when_epoch_finishes`

```python
when_epoch_finishes()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/events.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `Engine.when_epoch_starts`

```python
when_epoch_starts()
```






