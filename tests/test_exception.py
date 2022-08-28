import torchliter


def test_automated():

    import torch

    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise torchliter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise torchliter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise torchliter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = torchliter.engine.AutoEngine.build("MockEngine", train_step=classification)()

    try:
        eng.per_batch((0, 0))
    except torchliter.exception.ContinueIteration as e:
        assert isinstance(e, torchliter.exception.BadBatchError)
    try:
        eng.per_batch((-1, -1))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.FoundNanError)
        assert e.shutdown_engine

    try:
        eng.per_batch((-2, -2))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.BreakIteration)
        assert not e.shutdown_engine

    eng.trainloader = torch.utils.data.DataLoader(
        [(1, 1), (2, 2), (0, 0), (4, 4)], batch_size=1, shuffle=False
    )

    eng(torchliter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(torchliter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(torchliter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2


def test_automated_decorator():

    import torch

    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise torchliter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise torchliter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise torchliter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = torchliter.engine.AutoEngine.build("MockEngine", train_step=classification)()
    try:
        eng.per_batch((0, 0))
    except torchliter.exception.ContinueIteration as e:
        assert isinstance(e, torchliter.exception.BadBatchError)
    try:
        eng.per_batch((-1, -1))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.FoundNanError)
        assert e.shutdown_engine

    try:
        eng.per_batch((-2, -2))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.BreakIteration)
        assert not e.shutdown_engine

    eng.trainloader = torch.utils.data.DataLoader(
        [(1, 1), (2, 2), (0, 0), (4, 4)], batch_size=1, shuffle=False
    )

    eng(torchliter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(torchliter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(torchliter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2


def test_automated_decorator_config():

    import torch

    def classification(engine, batch):

        x, y = batch

        if x == 0 and y == 0:
            raise torchliter.exception.BadBatchError()

        if x == -1 and y == -1:
            raise torchliter.exception.FoundNanError()

        if x == -2 and y == -2:
            raise torchliter.exception.BreakIteration(shutdown_engine=False)

        yield "loss", 0.0

    eng = torchliter.engine.AutoEngine.build("MockEngine", train_step=classification)()
    try:
        eng.per_batch((0, 0))
    except torchliter.exception.ContinueIteration as e:
        assert isinstance(e, torchliter.exception.BadBatchError)
    try:
        eng.per_batch((-1, -1))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.FoundNanError)
        assert e.shutdown_engine

    try:
        eng.per_batch((-2, -2))
    except torchliter.exception.BreakIteration as e:
        assert isinstance(e, torchliter.exception.BreakIteration)
        assert not e.shutdown_engine

    eng.trainloader = torch.utils.data.DataLoader(
        [(1, 1), (2, 2), (0, 0), (4, 4)], batch_size=1, shuffle=False
    )

    eng(torchliter.stub.Train("trainloader")(2))

    assert eng.epoch == 2

    eng.reset_engine()

    eng.badloader = torch.utils.data.DataLoader(
        [(1, 1), (-1, -1), (1, 1)], batch_size=1, shuffle=False
    )
    eng(torchliter.stub.Train("badloader")(2))
    assert eng.epoch == 0 and eng.iteration == 1

    eng.reset_engine()

    eng.newloader = torch.utils.data.DataLoader([(1, 1), (1, 2), (-2, -2)])
    eng(torchliter.stub.Train("newloader")(2))
    assert eng.epoch == 2 and eng.iteration == 2
