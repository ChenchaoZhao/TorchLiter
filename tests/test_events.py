from liter.events import *


class MockEngine:

    epoch: int
    iteration: int

    def __init__(self, epoch=0, iteration=0):
        self.epoch = epoch
        self.iteration = iteration


def test_pre_epoch_handlers():

    ng = MockEngine()

    @PreEpochHandler.config(every=1)
    def every_epoch(engine):
        engine.iteration += 1

    assert isinstance(every_epoch, PreEpochHandler)

    for _ in range(10):
        every_epoch(ng)
        ng.epoch += 1

    assert ng.iteration == 10

    ng = MockEngine()

    @PreEpochHandler.config(every=2)
    def every_other_epoch(engine):
        engine.iteration += 1

    assert isinstance(every_other_epoch, PreEpochHandler)

    assert every_other_epoch.trigger_function

    for _ in range(10):
        every_other_epoch(ng)
        ng.epoch += 1

    assert ng.iteration == 5

    ng = MockEngine()

    @PreEpochHandler.config(
        trigger_function=lambda g: (g.epoch ** 2 % 5 == 0) and (g.epoch < 50)
    )
    def lambda_epoch(engine):
        engine.iteration = engine.epoch

    for _ in range(10):
        lambda_epoch(ng)
        ng.epoch += 1

    assert ng.iteration == 5


def test_post_epoch_handlers():

    ng = MockEngine()

    @PostEpochHandler.config(every=1)
    def every_epoch(engine):
        engine.iteration += 1

    assert isinstance(every_epoch, PostEpochHandler)

    for _ in range(10):
        ng.epoch += 1
        every_epoch(ng)

    assert ng.iteration == 10

    ng = MockEngine()

    @PostEpochHandler.config(every=2)
    def every_other_epoch(engine):
        engine.iteration += 1

    assert isinstance(every_other_epoch, PostEpochHandler)

    assert every_other_epoch.trigger_function

    for _ in range(10):
        ng.epoch += 1
        every_other_epoch(ng)

    assert ng.iteration == 5

    ng = MockEngine()

    @PreEpochHandler.config(
        trigger_function=lambda g: (g.epoch ** 2 % 5 == 0) and (g.epoch < 10)
    )
    def lambda_epoch(engine):
        engine.iteration = engine.epoch

    for _ in range(10):
        ng.epoch += 1
        lambda_epoch(ng)

    assert ng.iteration == 5


def test_pre_iteration_handlers():

    ng = MockEngine()

    @PreIterationHandler.config(every=1)
    def every_iter(engine):
        engine.epoch += 1

    assert isinstance(every_iter, PreIterationHandler)

    for _ in range(10):
        every_iter(ng)
        ng.iteration += 1

    assert ng.epoch == 10

    ng = MockEngine()

    @PreIterationHandler.config(every=2)
    def every_other_iter(engine):
        engine.epoch += 1

    assert isinstance(every_other_iter, PreIterationHandler)

    assert every_other_iter.trigger_function

    for _ in range(10):
        every_other_iter(ng)
        ng.iteration += 1

    assert ng.epoch == 5

    ng = MockEngine()

    @PreIterationHandler.config(
        trigger_function=lambda g: (g.iteration ** 2 % 5 == 0) and (g.iteration < 10)
    )
    def lambda_iter(engine):
        engine.epoch = engine.iteration

    for _ in range(10):
        lambda_iter(ng)
        ng.iteration += 1

    assert ng.epoch == 5