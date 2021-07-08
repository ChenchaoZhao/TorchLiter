<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.exception`






---

## <kbd>class</kbd> `BreakIteration`




### <kbd>method</kbd> `BreakIteration.__init__`

```python
__init__(shutdown_engine=False)
```









---

## <kbd>class</kbd> `ContinueIteration`








---

## <kbd>class</kbd> `BadBatchError`
Use case: 

current batch of data is corrupted, skip current batch and fetch a new batch and then continue. 





---

## <kbd>class</kbd> `StopEngine`
Use case: 

when raised at batch iteration level, if `shutdown_engine=True` then reraise BreakIteration to terminate engine. 

### <kbd>method</kbd> `StopEngine.__init__`

```python
__init__()
```









---

## <kbd>class</kbd> `GradientExplosionError`




### <kbd>method</kbd> `GradientExplosionError.__init__`

```python
__init__()
```









---

## <kbd>class</kbd> `FoundNanError`




### <kbd>method</kbd> `FoundNanError.__init__`

```python
__init__()
```









---

## <kbd>class</kbd> `EarlyStopping`




### <kbd>method</kbd> `EarlyStopping.__init__`

```python
__init__()
```









