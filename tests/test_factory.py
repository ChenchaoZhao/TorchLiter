from collections import namedtuple

import torch

import torchliter as lux


def test_repr_string():
    Struct = namedtuple("Struct", ["a", "b", "c"])
    s = Struct(1, 2, 5)
    rs = lux.factory.utils._repr_string(s)

    assert rs


def test_get_md5_hash():
    m = torch.nn.Linear(1, 2)
    l = torch.utils.data.DataLoader([i for i in range(10)])

    assert lux.factory.get_md5_hash(m)
    assert lux.factory.get_md5_hash(l)
    assert lux.factory.get_md5_hash(m) != lux.factory.get_md5_hash(l)


def test_registry():
    @lux.register_factory
    def make_model(in_features, out_features, bias=True):
        return torch.nn.Linear(in_features, out_features, bias=bias)

    assert "test_factory.make_model" in lux.factory.FACTORY_FUNCTION_REGISTRY

    m = make_model(1, 2)
    m_id = id(m)

    assert m_id in lux.factory.FACTORY_PRODUCT_REGISTRY

    assert isinstance(
        lux.factory.FACTORY_PRODUCT_REGISTRY[m_id], lux.factory.FactoryRecord
    )

    assert (
        lux.factory.FACTORY_PRODUCT_REGISTRY[m_id].factory_function_name
        == "test_factory.make_model"
    )
    assert lux.factory.FACTORY_PRODUCT_REGISTRY[m_id].input_parameters == dict(
        in_features=1, out_features=2, bias=True
    )

    cart = lux.Cart()
    cart.model = m
    assert (
        cart.attachment_records["model"] == lux.factory.FACTORY_PRODUCT_REGISTRY[m_id]
    )
    del cart.model
    assert "model" not in cart.attachment_records
