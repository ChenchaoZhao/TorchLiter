import inspect
import pickle

import torchliter


def test_sequence_container():
    container = torchliter.engine.buffers.SequenceContainer()
    assert container.values == []
    for i in range(5):
        container([i])
    assert container.values == [i for i in range(5)]
    container.reset()
    assert container.values == []


def test_scaler_buffer():
    scaler = torchliter.engine.buffers.ScalarSmoother(3)
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

    new_scaler = torchliter.engine.buffers.ScalarSmoother(3)
    new_scaler.load_state_dict(pickle.loads(state))

    assert new_scaler._count == 4
    assert new_scaler.mean == 2.0


def test_scalar_summary_statistics():
    b = torchliter.engine.buffers.ScalarSummaryStatistics()
    b(1.0)
    b(2.0)
    b(3.0)

    assert len(b) == 3
    assert b.median == 2.0
    assert b.min == 1.0
    assert b.max == 3.0
    assert b.mean == 2.0
    assert b.std == (2 / 3) ** (0.5)


def test_exponential_moving_average_buffer():
    scaler = torchliter.engine.buffers.ExponentialMovingAverage(1 / 3)

    assert scaler.mean is None
    assert scaler.variance == 0.0

    scaler(0)
    scaler(1)
    scaler(2)

    assert scaler.mean == 2 / 3 * (1 / 3 * 1) + 1 / 3 * 2

    scaler(3)
    state = scaler.state_dict()
    assert state["count"] == 4

    state = pickle.dumps(state)

    new_scaler = torchliter.engine.buffers.ExponentialMovingAverage(0.1)
    new_scaler.load_state_dict(pickle.loads(state))

    assert new_scaler._count == scaler._count
    assert new_scaler.mean == scaler.mean


class SimpleClass:
    def __init__(self):
        self.buffer = {
            f"arg{idx}": torchliter.engine.buffers.ScalarSmoother(5) for idx in range(3)
        }

    @torchliter.engine.utils.to_buffer("buffer")
    def generator(self):
        for idx in range(5):
            yield f"arg{idx}", idx


def test_decorator():
    def _generator(self):
        for idx in range(5):
            yield f"arg{idx}", idx

    assert inspect.isfunction(torchliter.engine.utils.to_buffer("buffer")(_generator))

    sc = SimpleClass()

    out = sc.generator()

    assert out is None

    for idx in range(3):
        assert sc.buffer[f"arg{idx}"].mean == idx
