<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.engine.base`




**Global Variables**
---------------
- **REPR_INDENT**
- **COMPONENTS**
- **map_str_to_types**
- **map_types_to_str**


---

## <kbd>class</kbd> `EngineBase`




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

#### <kbd>property</kbd> EngineBase.training







---

### <kbd>method</kbd> `EngineBase.after_iteration`

```python
after_iteration(**kwargs)
```





---

### <kbd>method</kbd> `EngineBase.before_iteration`

```python
before_iteration(**kwargs)
```





---

### <kbd>method</kbd> `EngineBase.eval`

```python
eval()
```





---

### <kbd>method</kbd> `EngineBase.execute`

```python
execute(**kwargs:Any)
```





---

### <kbd>method</kbd> `EngineBase.load_state_dict`

```python
load_state_dict(state_dict:Dict[str, Dict[str, Any]])
```





---

### <kbd>method</kbd> `EngineBase.per_batch`

```python
per_batch(batch:Union[Tuple[Any], Dict[str, Any]], **kwargs:Any)
```





---

### <kbd>method</kbd> `EngineBase.per_epoch`

```python
per_epoch(**kwargs)
```

Train model by one epoch. 

---

### <kbd>method</kbd> `EngineBase.queue`

```python
queue(stubs:List[liter.stub.StubBase])
```





---

### <kbd>method</kbd> `EngineBase.reset_engine`

```python
reset_engine()
```





---

### <kbd>method</kbd> `EngineBase.state_dict`

```python
state_dict() → Dict[str, Dict[str, Any]]
```





---

### <kbd>method</kbd> `EngineBase.train`

```python
train()
```





---

### <kbd>method</kbd> `EngineBase.when_epoch_finishes`

```python
when_epoch_finishes(**kwargs)
```





---

### <kbd>method</kbd> `EngineBase.when_epoch_starts`

```python
when_epoch_starts(**kwargs)
```






