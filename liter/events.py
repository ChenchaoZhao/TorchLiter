from enum import Enum
from typing import *


class EventCategory(Enum):

    EPOCH_STARTS = "epoch_starts"
    EPOCH_FINISHES = "epoch_finishes"
    BEFORE_ITERATION = "before_iteration"
    AFTER_ITERATION = "after_iteration"


class EventHandler:
    """Base Class for Event Handlers."""

    category: EventCategory

    def __init__(self, action_function: Optional[Callable] = None, **kwargs):
        self.action_function = action_function

    def trigger(self, *args, **kwargs) -> bool:
        return True

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
        return itertools.partial(cls, **kwargs)


class PreEpochHandler(EventHandler):
    category = EventCategory.EPOCH_STARTS


class PostEpochHandler(EventHandler):
    category = EventCategory.EPOCH_STARTS


class PreIterationHandler(EventHandler):
    category = EventCategory.BEFORE_ITERATION


class PostIterationHandler(EventHandler):
    category = EventCategory.AFTER_ITERATION
