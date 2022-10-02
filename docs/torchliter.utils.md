<!-- markdownlint-disable -->

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `torchliter.utils`




**Global Variables**
---------------
- **VALUE_TYPES**
- **LIST_TYPES**
- **DICT_TYPES**

---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/utils.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_object_from_module`

```python
get_object_from_module(module_path, object_name)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/utils.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `instantiate_class`

```python
instantiate_class(info: Dict)
```






---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/utils.py#L35"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_instance_from_dict`

```python
build_instance_from_dict(
    config: Dict,
    source_key: str,
    default_key: str = 'default_params'
)
```

Params:  config: Dict, config of the instance with keys  ('instance_name_1', 'instance_name_2', default_key) 

 source_key: str, name of the instance 

 default_key: str, name of the key of default params in config 


---

<a href="https://github.com/ChenchaoZhao/TorchLiter/tree/main/src/torchliter/utils.py#L88"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_progress_bar`

```python
get_progress_bar(itr: int, tot: int, width: int = 25)
```






