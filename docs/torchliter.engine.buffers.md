<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.engine.buffers`




**Global Variables**
---------------
- **REPR_INDENT**


---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BufferBase`
Buffer base class. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BufferBase.__init__`

```python
__init__(*args, **kwargs)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L39"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BufferBase.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BufferBase.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BufferBase.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BufferBase.update`

```python
update(x: Any)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SequenceContainer`
Sequence container Ingests new values and extends `self.value` 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SequenceContainer.__init__`

```python
__init__(*args, **kwargs)
```








---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L70"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SequenceContainer.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SequenceContainer.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SequenceContainer.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L63"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `SequenceContainer.update`

```python
update(sequence: Union[List, Tuple])
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ExponentialMovingAverage`
Exponential Moving Average of a series of Tensors. 

update rule:  EMA[x[t]] := (1 - alpha) * EMA[x[t-1]] + alpha * x[t] diff:  delta[x[t]] := x[t] - EMA[x[t-1]] 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ExponentialMovingAverage.__init__`

```python
__init__(alpha: float = 0.01, **kwargs: Any)
```






---

#### <kbd>property</kbd> ExponentialMovingAverage.std







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L115"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ExponentialMovingAverage.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L98"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ExponentialMovingAverage.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ExponentialMovingAverage.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ExponentialMovingAverage.update`

```python
update(x)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ScalarSummaryStatistics`
Store the scalars and compute statistics. 

The streaming scalars are stored in a list of any length. This is supposed to use in evals where the length is eval datasets. 

Available statistics: 
    - mean 
    - median 
    - std 
    - max 
    - min 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSummaryStatistics.__init__`

```python
__init__(**kwargs)
```






---

#### <kbd>property</kbd> ScalarSummaryStatistics.max





---

#### <kbd>property</kbd> ScalarSummaryStatistics.mean





---

#### <kbd>property</kbd> ScalarSummaryStatistics.median





---

#### <kbd>property</kbd> ScalarSummaryStatistics.min





---

#### <kbd>property</kbd> ScalarSummaryStatistics.std







---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSummaryStatistics.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSummaryStatistics.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSummaryStatistics.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSummaryStatistics.update`

```python
update(x: float)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ScalarSmoother`
Rolling average of a stream of scalars. 

The streaming scalars are stored in a deque of certain length (`maxlen`). The statistics are computed within the current deque. 

Available statistics: 
    - mean 
    - median 
    - std 
    - max 
    - min 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSmoother.__init__`

```python
__init__(window_size: int, **kwargs)
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

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSmoother.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSmoother.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSmoother.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L155"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `ScalarSmoother.update`

```python
update(x: float)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L232"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `VectorSmoother`
Exponential moving average of n-dim vector: 

vector = alpha * new_vector + (1 - alpha) * vector 

Additional features:  l_p normalization 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.__init__`

```python
__init__(
    alpha: float,
    n_dim: int,
    init_value: float,
    eps: float = 1e-08,
    normalize: bool = True,
    p: float = 1.0,
    device: str = 'cpu',
    dtype: torch.dtype = torch.float32,
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

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L289"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.load_state_dict`

```python
load_state_dict(state_dict)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L314"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.lp_norm`

```python
lp_norm(p: float)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L317"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.lp_normalized`

```python
lp_normalized(p: float)
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L273"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.reset`

```python
reset()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.state_dict`

```python
state_dict()
```





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/engine/buffers.py#L281"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `VectorSmoother.update`

```python
update(x: torch.Tensor)
```






