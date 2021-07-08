<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.engine.events`




**Global Variables**
---------------
- **REPR_INDENT**


---

## <kbd>class</kbd> `EventCategory`
An enumeration. 





---

## <kbd>class</kbd> `EventHandler`
Base Class for Event Handlers. 

### <kbd>method</kbd> `EventHandler.__init__`

```python
__init__(
    action_function:Optional[Callable[[liter.engine.base.EngineBase]], NoneType]=None,
    trigger_function:Optional[Callable[[liter.engine.base.EngineBase], bool]]=None,
    **kwargs
)
```








---

### <kbd>method</kbd> `EventHandler.action`

```python
action(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `EventHandler.config`

```python
config(*args, **kwargs)
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

### <kbd>method</kbd> `EventHandler.trigger`

```python
trigger(engine:liter.engine.base.EngineBase) → bool
```






---

## <kbd>class</kbd> `PreEpochHandler`
Hanldes events when a new epoch starts. 

### <kbd>method</kbd> `PreEpochHandler.__init__`

```python
__init__(
    action_function:Callable[[liter.engine.base.EngineBase], NoneType],
    trigger_function:Optional[Callable[[liter.engine.base.EngineBase], bool]]=None,
    every:int=1
)
```








---

### <kbd>method</kbd> `PreEpochHandler.action`

```python
action(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `PreEpochHandler.config`

```python
config(*args, **kwargs)
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

### <kbd>method</kbd> `PreEpochHandler.trigger`

```python
trigger(engine:liter.engine.base.EngineBase) → bool
```






---

## <kbd>class</kbd> `PostEpochHandler`
Hanldes events when a new epoch finishes. 

### <kbd>method</kbd> `PostEpochHandler.__init__`

```python
__init__(
    action_function:Callable[[liter.engine.base.EngineBase], NoneType],
    trigger_function:Optional[Callable[[liter.engine.base.EngineBase], bool]]=None,
    every:int=1
)
```








---

### <kbd>method</kbd> `PostEpochHandler.action`

```python
action(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `PostEpochHandler.config`

```python
config(*args, **kwargs)
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

### <kbd>method</kbd> `PostEpochHandler.trigger`

```python
trigger(engine:liter.engine.base.EngineBase) → bool
```






---

## <kbd>class</kbd> `PreIterationHandler`
Handles events before each iteration. 

### <kbd>method</kbd> `PreIterationHandler.__init__`

```python
__init__(
    action_function:Callable[[liter.engine.base.EngineBase], NoneType],
    trigger_function:Optional[Callable[[liter.engine.base.EngineBase], bool]]=None,
    every:int=1
)
```








---

### <kbd>method</kbd> `PreIterationHandler.action`

```python
action(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `PreIterationHandler.config`

```python
config(*args, **kwargs)
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

### <kbd>method</kbd> `PreIterationHandler.trigger`

```python
trigger(engine:liter.engine.base.EngineBase) → bool
```






---

## <kbd>class</kbd> `PostIterationHandler`
Handles events after each iteration. 

### <kbd>method</kbd> `PostIterationHandler.__init__`

```python
__init__(
    action_function:Callable[[liter.engine.base.EngineBase], NoneType],
    trigger_function:Optional[Callable[[liter.engine.base.EngineBase], bool]]=None,
    every:int=1
)
```








---

### <kbd>method</kbd> `PostIterationHandler.action`

```python
action(*args, **kwargs)
```





---

### <kbd>classmethod</kbd> `PostIterationHandler.config`

```python
config(*args, **kwargs)
```

Add additional config kwargs. 

For example ==================== 

@EventHandler.config(param1=1.0, param2=2.0) def some_action_function(engine):  ... 

---

### <kbd>method</kbd> `PostIterationHandler.trigger`

```python
trigger(engine:liter.engine.base.EngineBase) → bool
```






---

## <kbd>class</kbd> `Engine`
Engine with Event Handler plugin. 

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

#### <kbd>property</kbd> Engine.training







---

### <kbd>method</kbd> `Engine.after_iteration`

```python
after_iteration()
```





---

### <kbd>method</kbd> `Engine.attach_event`

```python
attach_event(handler:liter.engine.events.EventHandler)
```





---

### <kbd>method</kbd> `Engine.before_iteration`

```python
before_iteration()
```





---

### <kbd>method</kbd> `Engine.list_events`

```python
list_events(event_category:Optional[str]=None)
```





---

### <kbd>method</kbd> `Engine.when_epoch_finishes`

```python
when_epoch_finishes()
```





---

### <kbd>method</kbd> `Engine.when_epoch_starts`

```python
when_epoch_starts()
```






