import liter


def test_automated():

    import torch

    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise liter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise liter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise liter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = liter.engine.Automated.from_forward(classification)
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

    eng(liter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(liter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(liter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2


def test_automated_decorator():

    import torch

    @liter.engine.Automated
    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise liter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise liter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise liter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = classification
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

    eng(liter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(liter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(liter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2


def test_automated_decorator_config():

    import torch

    @liter.engine.Automated.config(smooth_window=100)
    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise liter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise liter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise liter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = classification
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

    eng(liter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(liter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(liter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2
