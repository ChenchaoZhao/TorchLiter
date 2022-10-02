<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.engine.utils`





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/utils.py#L10"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `to_buffer`

```python
to_buffer(buffer_registry_name='buffer_registry') → Callable
```

Returns a decorator that push the updates to corresponding buffers. 

For example, 

```
@to_buffer('some-buffer-registry'):
def some_step_method(self, *args):
     ...
     yield 'var1', var1
     ...
     yield 'var2', var2
``` where `var1` and `var2` are buffer names in `some-buffer-registry`. 



Parameters 
---------- buffer_registry_name : str, optional  name of buffer registry, by default "buffer_registry" 

Returns 
------- Callable  A decorator that turns a generator to a method the automatically  pushes updates to buffers 


