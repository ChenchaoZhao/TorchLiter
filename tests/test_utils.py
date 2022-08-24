from torchliter.utils import _convert_str_to_py_object_name, _is_py_object_name


def test_py_object_name():

    names = ["aBc", " a Bc", "@bc", "__b_"]

    is_py_name = [True, False, False, True]

    converted = ["aBc", "a_Bc", "_bc", "__b_"]

    for name, res, conv in zip(names, is_py_name, converted):
        assert _is_py_object_name(name) == res
        assert _convert_str_to_py_object_name(name) == conv
