<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.exception`
Exceptions used by the `Engine` class. 



---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L4"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BreakIteration`
BreakIteration Exception. 

Parameters 
---------- shutdown_engine : bool  If flag variable `shutdown_engine` (the default is False) is `True`,  the `Engine` object will kill both the interation and the epochs 

Attributes 
---------- shutdown_engine 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `BreakIteration.__init__`

```python
__init__(shutdown_engine: bool = False)
```









---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ContinueIteration`
ContinueIteration Exception. 

When raised, the `Engine` iteration will continue. 





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L31"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BadBatchError`
BadBatchError Exception, subclass of ContinueIteration. 

If current batch of data is corrupted, skip current batch and fetch a new batch and then continue. 





---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `StopEngine`
StopEngine Exception, subclass of BreakIteration. 

When raised at batch iteration level, if `shutdown_engine=True` then reraise BreakIteration to terminate engine. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `StopEngine.__init__`

```python
__init__()
```









---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `GradientExplosionError`
GradientExplosionError, subclass of StopEngine. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `GradientExplosionError.__init__`

```python
__init__()
```









---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FoundNanError`
FoundNanError, subclass of StopEngine. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `FoundNanError.__init__`

```python
__init__()
```









---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EarlyStopping`
EarlyStopping, subclass of StopEngine. 

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/exception.py#L48"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `EarlyStopping.__init__`

```python
__init__()
```









