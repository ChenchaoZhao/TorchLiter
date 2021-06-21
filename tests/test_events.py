from collections import namedtuple

from liter.events import *

MockEngine = namedtuple("MockEngine", ["epoch", "iteration"])


def test_epoch_handlers():

    ng = MockEngine(epoch=0, iteration=0)

    @PreEpochHandler.config(every=1)
    def pre_epoch(engine):
        engine.iteration += 1

    for _ in range(10):
        pre_epoch(ng)

    assert ng.iteration == 10
