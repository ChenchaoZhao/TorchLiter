<!-- markdownlint-disable -->

# <kbd>module</kbd> `liter.stub`




**Global Variables**
---------------
- **REPR_INDENT**


---

## <kbd>class</kbd> `StubBase`
Base class for Stubs. 

### <kbd>method</kbd> `StubBase.__init__`

```python
__init__(*args, **kwargs)
```








---

### <kbd>method</kbd> `StubBase.replicate`

```python
replicate(copy:int=1)
```






---

## <kbd>class</kbd> `Train`
Train stub. 

### <kbd>method</kbd> `Train.__init__`

```python
__init__(dataloader:str, iteration:int=0, **kwargs)
```








---

### <kbd>method</kbd> `Train.replicate`

```python
replicate(copy:int=1)
```






---

## <kbd>class</kbd> `Evaluate`
Evaluation stub. 

### <kbd>method</kbd> `Evaluate.__init__`

```python
__init__(dataloader:str, **kwargs)
```








---

### <kbd>method</kbd> `Evaluate.replicate`

```python
replicate(copy:int=1)
```






---

## <kbd>class</kbd> `Lambda`
General action stub. 

### <kbd>method</kbd> `Lambda.__init__`

```python
__init__(action:str, **kwargs)
```








---

### <kbd>method</kbd> `Lambda.replicate`

```python
replicate(copy:int=1)
```






