from enum import Enum
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type

from .. import REPR_INDENT
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
        **kwargs,
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

    def __repr__(self):
        lines = []
        lines.append(self.__class__.__name__)
        lines.append(" " * REPR_INDENT + "trigger function:")
        if self.trigger_function:
            lines.append(" " * REPR_INDENT * 2 + self.trigger_function.__name__)
        lines.append(" " * REPR_INDENT + "action function:")
        if self.action_function:
            lines.append(" " * REPR_INDENT * 2 + self.action_function.__name__)

        return "\n".join(lines)

    @classmethod
    def config(cls, *args, **kwargs) -> Type:
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

    def _default_trigger(
        self,
        engine: EngineBase,
        level: str,
        every: int = 1,
        train_stub: bool = True,
        eval_stub: bool = True,
        lambda_stub: bool = False,
    ) -> bool:
        """
        Default trigger function for all `EventHandler`s.

        only trigger when training: train_stub=True, eval_stub=False, lambda_stub=False
        only trigger when eval: train_stub=False, eval_stub=False, lambda_stub=False

        Parameters
        ----------
        engine : EngineBase
            positional arg for engine
        level : str
            either 'epoch' or 'iteration'
        every : int, optional
            every 'level', by default 1
        train_stub : bool, optional
            trigger event at train stub if True, by default True
        eval_stub : bool, optional
            trigger event at eval stub if True, by default True
        lambda_stub : bool, optional
            trigger event at lambda stub if True, by default False

        Returns
        -------
        bool
            whether or not trigger the event
        """
        if level == "epoch":
            every_flag = int(engine.epoch) % int(every) == 0
        elif level == "interation":
            every_flag = int(engine.iteration) % int(every) == 0
        else:
            raise ValueError(f"level can be `epoch` or `iteration` but got `{level}`.")

        if not every_flag:
            return False

        if not train_stub and engine.is_train_stub:
            return False

        if not eval_stub and engine.is_eval_stub:
            return False

        if not lambda_stub and engine.is_lambda_stub:
            return False

        return True

    def _default_epoch_trigger(
        self,
        engine: EngineBase,
        every: int = 1,
        train_stub: bool = True,
        eval_stub: bool = True,
        lambda_stub: bool = False,
    ) -> bool:
        return self._default_trigger(
            engine,
            level="epoch",
            every=every,
            train_stub=train_stub,
            eval_stub=eval_stub,
            lambda_stub=lambda_stub,
        )

    def _default_iteration_trigger(
        self,
        engine: EngineBase,
        every: int = 1,
        train_stub: bool = True,
        eval_stub: bool = True,
        lambda_stub: bool = False,
    ) -> bool:
        return self._default_trigger(
            engine,
            level="iteration",
            every=every,
            train_stub=train_stub,
            eval_stub=eval_stub,
            lambda_stub=lambda_stub,
        )


class PreEpochHandler(EventHandler):
    """Hanldes events when a new epoch starts."""

    category = EventCategory.EPOCH_STARTS

    def __init__(
        self,
        action_function: Callable[[EngineBase], None],
        trigger_function: Optional[Callable[[EngineBase], bool]] = None,
        every: int = 1,
        train_stub: bool = True,
        eval_stub: bool = True,
        lambda_stub: bool = False,
    ):
        if trigger_function is None:

            def trigger_function(engine: EngineBase):
                return self._default_epoch_trigger(
                    engine, every, train_stub, eval_stub, lambda_stub
                )

        super().__init__(action_function, trigger_function)


class PostEpochHandler(EventHandler):
    """Hanldes events when a new epoch finishes."""

    category = EventCategory.EPOCH_FINISHES

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
    """
    Engine with Event Handler plugin.

    Attributes
    ----------
    _event_handlers : Dict[EventCategory, List[EventHandler]]
        The registry of event handlers.
    """

    _event_handlers: Dict[EventCategory, List[EventHandler]]

    def __init__(self):
        super().__init__()
        self._event_handlers = {
            EventCategory.EPOCH_STARTS: [],
            EventCategory.EPOCH_FINISHES: [],
            EventCategory.BEFORE_ITERATION: [],
            EventCategory.AFTER_ITERATION: [],
        }

    def attach_event(self, handler: EventHandler):
        """
        Attach an event handler.

        Parameters
        ----------
        handler : EventHandler
            An event handler to be attached.
        """
        if isinstance(handler, EventHandler) and handler.category:
            self._event_handlers[handler.category].append(handler)
        else:
            raise TypeError("Category of handler must be specified.")

    def list_events(self, event_category: Optional[str] = None) -> Tuple[EventHandler]:
        """
        List events based on category.

        Parameters
        ----------
        event_category : Optional[str]
            List event handlers in `event_category` (the default is None).
            If not provided, list all handlers.

        Returns
        -------
        Tuple[EventHandler]
            Tuple of handlers.
        """
        return (
            self._event_handlers[EventCategory(event_category)]
            if event_category
            else self._event_handlers
        )

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
