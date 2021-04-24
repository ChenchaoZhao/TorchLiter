import pickle

import liter


def test_scaler_buffer():

    scaler = liter.buffer.ScalarSmoother(3)

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
