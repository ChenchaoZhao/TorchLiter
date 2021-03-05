import abc

from .common import REPR_INDENT


class StubBase(abc.ABC):
    """Base class for Stubs"""

    def __init__(self, *args, **kwargs):
        assert len(args) == 0, \
        "There should not be any args only kwargs allowed."
        for k, v in kwargs.items():
            setattr(self, k, v)

    def replicate(self, copy: int = 1):
        # make copies of stubs with same kwargs
        copy = int(copy)
        assert copy > 0
        kwargs = self.__dict__
        copies = []
        for _ in range(copy):
            copies.append(self.__class__(**kwargs))
        return copies

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        for k, v in self.__dict__.items():
            if k.startswith('_'):
                continue
            out.append(' ' * REPR_INDENT + f"{k}: {v}")

        return '\n'.join(out)


class StubTrain(StubBase):
    """Train Stub"""

    def __init__(self, dataloader: str, **kwargs):

        assert isinstance(dataloader, str), \
        f"Dataloader should be string type but get type {type(dataloader)}"

        kwargs['dataloader'] = dataloader
        kwargs['action'] = 'train'
        kwargs['epoch'] = 1

        super().__init__(**kwargs)


class StubEval(StubBase):
    """Evaluation Stub"""

    def __init__(self, dataloader: str, **kwargs):

        assert isinstance(dataloader, str), \
        f"Dataloader should be string type but get type {type(dataloader)}"
        kwargs['dataloader'] = dataloader
        kwargs['action'] = 'evaluate'
        kwargs['epoch'] = 1

        super().__init__(**kwargs)
