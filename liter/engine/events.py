from enum import Enum
from functools import partial
from typing import *

from .base import EngineBase

__all__ = [
    "EventCategory",
    "EventHandler",
    "PreEpochHandler",
    "PostEpochHandler",
    "PreIterationHandler",
    "PostIterationHandler",
    "Engine",
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
        action_function: Optional[Callable[[EngineBase], None]] = None,
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
        **kwargs
    ):
        self.action_function = action_function
        self.trigger_function = trigger_function

    def trigger(self, engine: EngineBase) -> bool:
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
        action_function: Callable[[EngineBase], None],
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
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
        action_function: Callable[[EngineBase], None],
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
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
        action_function: Callable[[EngineBase], None],
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
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
        action_function: Callable[[EngineBase], None],
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
        every: int = 1,
    ):
        if trigger_function is None:
            trigger_function = lambda engine: engine.iteration % every == 0  # noqa
        super().__init__(action_function, trigger_function)


class Engine(EngineBase):
    """Engine with Event Handler plugin."""

    _event_handlers: Dict[EventCategory, List[EventHandler]] = {
        EventCategory.EPOCH_STARTS: [],
        EventCategory.EPOCH_FINISHES: [],
        EventCategory.BEFORE_ITERATION: [],
        EventCategory.AFTER_ITERATION: [],
    }

    def attach_event(self, handler: EventHandler):
        if isinstance(handler, EventHandler) and handler.category:
            self._event_handlers[handler.category].append(handler)
        else:
            raise TypeError("Category of handler must be specified.")

    def when_epoch_starts(self):
        for h in self._event_handlers[EventCategory.EPOCH_STARTS]:
            h(self)

    def when_epoch_finishes(self):
        for h in self._event_handlers[EventCategory.EPOCH_FINISHES]:
            h(self)

    def before_iteration(self):
        for h in self._event_handlers[EventCategory.BEFORE_ITERATION]:
            h(self)

    def after_iteration(self):
        for h in self._event_handlers[EventCategory.AFTER_ITERATION]:
            h(self)
