import torch
import liter


def test_automated():

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise liter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise liter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise liter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = liter.Automated.from_forward(classification)
    try:
        eng.per_batch((0, 0))
    except liter.exception.ContinueIteration as e:
        assert isinstance(e, liter.exception.BadBatchError)
    try:
        eng.per_batch((-1, -1))
    except liter.exception.BreakIteration as e:
        assert isinstance(e, liter.exception.FoundNanError)
        assert e.shutdown_engine

    try:
        eng.per_batch((-2, -2))
    except liter.exception.BreakIteration as e:
        assert isinstance(e, liter.exception.BreakIteration)
        assert not e.shutdown_engine

    eng.trainloader = torch.utils.data.DataLoader(
        [(1, 1), (2, 2), (0, 0), (4, 4)], batch_size=1, shuffle=False
    )

    eng(liter.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(liter.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(liter.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2
