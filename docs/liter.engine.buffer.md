<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.engine.buffer`




**Global Variables**
---------------
- **REPR_INDENT**

---

## <kbd>function</kbd> `to_buffer`

```python
to_buffer(name='buffer_registry')
```






---

## <kbd>class</kbd> `BufferBase`
Buffer base class. 

### <kbd>method</kbd> `BufferBase.__init__`

```python
__init__(*args, **kwargs)
```








---

### <kbd>method</kbd> `BufferBase.load_state_dict`

```python
load_state_dict(state_dict)
```





---

### <kbd>method</kbd> `BufferBase.reset`

```python
reset()
```





---

### <kbd>method</kbd> `BufferBase.state_dict`

```python
state_dict()
```





---

### <kbd>method</kbd> `BufferBase.update`

```python
update(x:Any)
```






---

## <kbd>class</kbd> `ExponentialMovingAverage`
Exponential Moving Average of a series of Tensors. 

update rule:  EMA[x[t]] := (1 - alpha) * EMA[x[t-1]] + alpha * x[t] diff:  delta[x[t]] := x[t] - EMA[x[t-1]] 

### <kbd>method</kbd> `ExponentialMovingAverage.__init__`

```python
__init__(
    alpha:Optional[float]=None,
    window_size:Optional[int]=None,
    **kwargs:Any
)
```






---

#### <kbd>property</kbd> ExponentialMovingAverage.std







---

### <kbd>method</kbd> `ExponentialMovingAverage.load_state_dict`

```python
load_state_dict(state_dict)
```





---

### <kbd>method</kbd> `ExponentialMovingAverage.reset`

```python
reset()
```





---

### <kbd>method</kbd> `ExponentialMovingAverage.state_dict`

```python
state_dict()
```





---

### <kbd>method</kbd> `ExponentialMovingAverage.update`

```python
update(x)
```






---

## <kbd>class</kbd> `ScalarSmoother`
Rolling smoothing buffer for scalars. 

### <kbd>method</kbd> `ScalarSmoother.__init__`

```python
__init__(window_size:int, **kwargs)
```






---

#### <kbd>property</kbd> ScalarSmoother.max





---

#### <kbd>property</kbd> ScalarSmoother.mean





---

#### <kbd>property</kbd> ScalarSmoother.median





---

#### <kbd>property</kbd> ScalarSmoother.min





---

#### <kbd>property</kbd> ScalarSmoother.std







---

### <kbd>method</kbd> `ScalarSmoother.load_state_dict`

```python
load_state_dict(state_dict)
```





---

### <kbd>method</kbd> `ScalarSmoother.reset`

```python
reset()
```





---

### <kbd>method</kbd> `ScalarSmoother.state_dict`

```python
state_dict()
```





---

### <kbd>method</kbd> `ScalarSmoother.update`

```python
update(x:float)
```






---

## <kbd>class</kbd> `VectorSmoother`
Exponential moving average of n-dim vector: 

vector = alpha * new_vector + (1 - alpha) * vector 

Additional features:  l_p normalization 

### <kbd>method</kbd> `VectorSmoother.__init__`

```python
__init__(
    alpha:float,
    n_dim:int,
    init_value:float,
    eps:float=1e-08,
    normalize:bool=True,
    p:float=1.0,
    device:str='cpu',
    dtype:torch.dtype=torch.float32,
    **kwargs
)
```






---

#### <kbd>property</kbd> VectorSmoother.l1_norm





---

#### <kbd>property</kbd> VectorSmoother.l1_normalized





---

#### <kbd>property</kbd> VectorSmoother.l2_norm





---

#### <kbd>property</kbd> VectorSmoother.l2_normalized





---

#### <kbd>property</kbd> VectorSmoother.std





---

#### <kbd>property</kbd> VectorSmoother.vector







---

### <kbd>method</kbd> `VectorSmoother.load_state_dict`

```python
load_state_dict(state_dict)
```





---

### <kbd>method</kbd> `VectorSmoother.lp_norm`

```python
lp_norm(p:float)
```





---

### <kbd>method</kbd> `VectorSmoother.lp_normalized`

```python
lp_normalized(p:float)
```





---

### <kbd>method</kbd> `VectorSmoother.reset`

```python
reset()
```





---

### <kbd>method</kbd> `VectorSmoother.state_dict`

```python
state_dict()
```





---

### <kbd>method</kbd> `VectorSmoother.update`

```python
update(x:torch.Tensor)
```






