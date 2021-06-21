from enum import Enum
from functools import partial
from typing import *

Engine = TypeVar("Engine")

__all__ = [
    "EventCategory",
    "EventHandler",
    "PreEpochHandler",
    "PostEpochHandler",
    "PreIterationHandler",
    "PostIterationHandler",
]


class EventCategory(Enum):

    EPOCH_STARTS = "epoch_starts"
    EPOCH_FINISHES = "epoch_finishes"
    BEFORE_ITERATION = "before_iteration"
    AFTER_ITERATION = "after_iteration"


class EventHandler:
    """Base Class for Event Handlers."""

    category: EventCategory

    def __init__(
        self,
        action_function: Optional[Callable[[Engine], None]] = None,
        trigger_function: Optional[Callable[[Engine], bool]] = None,
        **kwargs
    ):
        self.action_function = action_function
        self.trigger_function = trigger_function

    def trigger(self, engine: Engine) -> bool:
        if self.trigger_function is None:
            return True

        return self.trigger_function(engine)

    def action(self, *args, **kwargs):
        if self.action_function is None:
            raise NotImplementedError(
                "Method `action` must be implemented if action_function is not provided."
            )
        else:
            self.action_function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        if self.trigger(*args, **kwargs):
            self.action(*args, **kwargs)

    @classmethod
    def config(cls, *args, **kwargs):
        """
        Add additional config kwargs.

        For example
        ====================

        @EventHandler.config(param1=1.0, param2=2.0)
        def some_action_function(engine):
            ...
        """
        if args:
            raise ValueError("Only kwargs are allowed.")
        return partial(cls, **kwargs)


class PreEpochHandler(EventHandler):
    """Hanldes events when a new epoch starts."""

    category = EventCategory.EPOCH_STARTS

    def __init__(
        self,
        action_function: Callable[[Engine], None],
        trigger_function: Optional[Callable[[Engine], bool]] = None,
        every: int = 1,
    ):
        if trigger_function is None:
            trigger_function = lambda engine: engine.epoch % every == 0  # noqa
        super().__init__(action_function, trigger_function)


class PostEpochHandler(EventHandler):
    """Hanldes events when a new epoch finishes."""

    category = EventCategory.EPOCH_STARTS

    def __init__(
        self,
        action_function: Callable[[Engine], None],
        trigger_function: Optional[Callable[[Engine], bool]] = None,
        every: int = 1,
    ):
        if trigger_function is None:
            trigger_function = lambda engine: engine.epoch % every == 0  # noqa
        super().__init__(action_function, trigger_function)


class PreIterationHandler(EventHandler):
    """Handles events before each iteration."""

    category = EventCategory.BEFORE_ITERATION

    def __init__(
        self,
        action_function: Callable[[Engine], None],
        trigger_function: Optional[Callable[[Engine], bool]] = None,
        every: int = 1,
    ):
        if trigger_function is None:
            trigger_function = lambda engine: engine.iteration % every == 0  # noqa
        super().__init__(action_function, trigger_function)


class PostIterationHandler(EventHandler):
    """Handles events after each iteration."""

    category = EventCategory.AFTER_ITERATION

    def __init__(
        self,
        action_function: Callable[[Engine], None],
        trigger_function: Optional[Callable[[Engine], bool]] = None,
        every: int = 1,
    ):
        if trigger_function is None:
            trigger_function = lambda engine: engine.iteration % every == 0  # noqa
        super().__init__(action_function, trigger_function)
