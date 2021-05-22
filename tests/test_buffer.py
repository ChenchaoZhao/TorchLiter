import pickle
import torch
import liter


def test_scaler_buffer():

    scaler = liter.buffer.ScalarSmoother(3)
    print(scaler)

    assert scaler.mean == 0.0
    assert scaler.max == 0.0
    assert scaler.std == 0.0
    assert scaler.median == 0.0

    scaler(0)
    scaler(1)
    scaler(2)

    assert scaler.mean == 1.0

    scaler(3)
    assert scaler.mean == 2.0

    state = scaler.state_dict()
    assert state["count"] == 4

    state = pickle.dumps(state)

    new_scaler = liter.buffer.ScalarSmoother(3)
    new_scaler.load_state_dict(pickle.loads(state))

    assert new_scaler._count == 4
    assert new_scaler.mean == 2.0


def test_vector_buffer():

    vector = liter.buffer.VectorSmoother(0.5, 8, 2.0)
    print(vector)

    assert vector.l1_norm == 2.0 * 8
    assert vector.l2_norm == (2.0 ** 2 * 8) ** 0.5
    assert (vector.l1_normalized == 1 / 8 * torch.ones(8, dtype=float)).all()
    assert (vector.l2_normalized == (1 / 8 ** 0.5) * torch.ones(8, dtype=float)).all()

    vector(torch.zeros(8))
    assert (vector.vector == 2 * torch.ones(8) * 0.5).all()
    vector(torch.zeros(8))
    assert (vector.vector == 2 * torch.ones(8) * 0.5 ** 2).all()
    vector(torch.zeros(8))
    assert (vector.vector == 2 * torch.ones(8) * 0.5 ** 3).all()

    assert vector._count == 3

    state = vector.state_dict()
    assert state["count"] == 3

    state = pickle.dumps(state)

    new_vector = liter.buffer.VectorSmoother(0.5, 8, 2.0)
    new_vector.load_state_dict(pickle.loads(state))
    assert new_vector._count == 3
    assert (new_vector._state == 2 * torch.ones(8) * 0.5 ** 3).all()


class SimpleClass:
    def __init__(self):
        self.buffer = {f"arg{idx}": liter.buffer.ScalarSmoother(5) for idx in range(3)}

    @liter.buffer.to_buffer("buffer")
    def generator(self):
        for idx in range(5):
            yield f"arg{idx}", idx


def test_decorator():

    sc = SimpleClass()

    out = sc.generator()

    assert out == None

    for idx in range(3):
        assert sc.buffer[f"arg{idx}"].mean == idx
