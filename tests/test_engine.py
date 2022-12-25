import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchliter


class SimpleEngine(torchliter.engine.EngineBase):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Linear(1, 3)
        self.dataloader = torch.utils.data.DataLoader(
            torch.randn(1000, 1), batch_size=10
        )
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)
        self.gradscaler = torch.cuda.amp.GradScaler()

        self.total_iteration = 0

    def per_batch(self, batch):

        out = self.model(batch)

        loss = out.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.total_iteration += 1

    def when_epoch_starts(self):
        pass

    def when_epoch_finishes(self):
        pass


def test_engine_base():
    trainer = SimpleEngine()

    print(trainer)

    assert "model" in trainer.model_registry
    assert "scheduler" in trainer.scheduler_registry
    assert "gradscaler" in trainer.gradscaler_registry

    trainer(torchliter.stub.Train("dataloader")(2))

    assert trainer.epoch == 2
    assert trainer.iteration == 1000 // 10
    assert trainer.total_iteration == 2000 // 10

    state_dict = trainer.state_dict()

    trainer.load_state_dict(state_dict)


def test_cart():
    cart = torchliter.engine.auto.Cart(
        model=torch.nn.Linear(1, 1),
        train_loader=torch.utils.data.DataLoader([i for i in range(100)], batch_size=1),
    )

    assert str(cart)
    assert str(cart) == repr(cart)
    cart.attach(
        eval_loader=torch.utils.data.DataLoader([i for i in range(100)], batch_size=1)
    )
    cart.eval_loader2 = torch.utils.data.DataLoader(
        [i for i in range(100)], batch_size=1
    )
    assert cart.eval_loader.__class__ == cart.eval_loader2.__class__

    del cart.eval_loader2

    cart.optimizer = torch.optim.Adam(cart.model.parameters(), lr=0.1)
    _types = dict(
        model=torch.nn.Module,
        train_loader=torch.utils.data.DataLoader,
        eval_loader=torch.utils.data.DataLoader,
        optimizer=torch.optim.Optimizer,
    )
    for var, obj in cart.kwargs.items():
        assert isinstance(obj, _types[var])


def test_auto_buffers():
    def train_step(_, batch):
        yield "test 1", 1.0
        yield "test 2", 2.0
        yield "test 3", 3.0

    dict_of_buffers = torchliter.engine.AutoEngine.auto_buffers(
        train_step, torchliter.engine.buffers.ExponentialMovingAverage, alpha=1 / 314
    )
    for var in ["test_1", "test_2", "test_3"]:
        assert var in dict_of_buffers
        b = dict_of_buffers[var]
        assert isinstance(b, torchliter.engine.buffers.ExponentialMovingAverage)
        assert b.alpha == 1.0 / 314.0

    dict_of_buffers = torchliter.engine.AutoEngine.auto_buffers(
        train_step, torchliter.engine.buffers.ScalarSummaryStatistics
    )
    for var in ["test_1", "test_2", "test_3"]:
        assert var in dict_of_buffers
        b = dict_of_buffers[var]
        assert isinstance(b, torchliter.engine.buffers.ScalarSummaryStatistics)
        assert b.maxlen is None

    # Using Cart.parse_buffers
    cart = torchliter.engine.auto.Cart()
    cart.parse_buffers(train_step, mode="train")
    for var in ["test_1", "test_2", "test_3"]:
        assert var in cart.kwargs
        b = cart.kwargs[var]
        assert isinstance(b, torchliter.engine.buffers.ExponentialMovingAverage)

    cart = torchliter.engine.auto.Cart()
    cart.parse_buffers(train_step, mode="eval")
    for var in ["test_1", "test_2", "test_3"]:
        assert var in cart.kwargs
        b = cart.kwargs[var]
        assert isinstance(b, torchliter.engine.buffers.ScalarSummaryStatistics)


def test_auto_engine():
    cart = torchliter.engine.auto.Cart()
    cart.model = nn.Linear(1, 10)
    cart.train_loader = torch.utils.data.DataLoader(
        [(torch.randn(1), i) for i in range(10)], batch_size=5
    )
    cart.eval_loader = torch.utils.data.DataLoader(
        [(torch.randn(1), i) for i in range(10)], batch_size=5
    )
    cart.optimizer = torch.optim.AdamW(
        cart.model.parameters(), lr=1e-3, weight_decay=1e-5
    )

    def train_step(_, batch, **kwargs):
        image, target = batch
        logits = _.model(image)
        loss = F.cross_entropy(logits, target)
        _.optimizer.zero_grad()
        loss.backward()
        _.optimizer.step()

        yield "cross entropy loss", loss.item()

        acc = (logits.max(-1).indices == target).float().mean()

        yield "train acc", acc.item()

        yield "train kwarg", kwargs["example"]

    def eval_step(_, batch, **kwargs):
        image, target = batch
        with torch.no_grad():
            logits = _.model(image)
        acc = (logits.max(-1).indices == target).float().mean()
        yield "eval acc", acc.item()

        yield "eval kwarg", kwargs["example"]

    def hello(_):
        print("hello")

    train_buffers = torchliter.engine.AutoEngine.auto_buffers(
        train_step, torchliter.engine.buffers.ExponentialMovingAverage
    )
    eval_buffers = torchliter.engine.AutoEngine.auto_buffers(
        eval_step, torchliter.engine.buffers.ScalarSummaryStatistics
    )
    TestEngineClass = torchliter.engine.AutoEngine.build(
        "TestEngine", train_step, eval_step, print_hello=hello
    )
    test_engine = TestEngineClass(**{**cart.kwargs, **train_buffers, **eval_buffers})

    assert isinstance(test_engine, torchliter.engine.AutoEngine)

    assert inspect.ismethod(test_engine.print_hello)
    assert inspect.ismethod(test_engine.train_step)
    assert inspect.ismethod(test_engine.eval_step)
    assert inspect.isgeneratorfunction(test_engine._train_step_generator)
    assert inspect.isgeneratorfunction(test_engine._eval_step_generator)

    assert isinstance(test_engine.model, nn.Linear)
    assert isinstance(test_engine.train_loader, torch.utils.data.DataLoader)
    assert isinstance(test_engine.eval_loader, torch.utils.data.DataLoader)
    assert isinstance(test_engine.optimizer, torch.optim.Optimizer)

    assert isinstance(
        test_engine.train_acc, torchliter.engine.buffers.ExponentialMovingAverage
    )
    assert isinstance(
        test_engine.cross_entropy_loss,
        torchliter.engine.buffers.ExponentialMovingAverage,
    )
    assert isinstance(
        test_engine.eval_acc, torchliter.engine.buffers.ScalarSummaryStatistics
    )

    test_engine(
        torchliter.stub.Train("train_loader")(1)
        + torchliter.stub.Evaluate("eval_loader")(1),
        example=0.618,
    )

    assert test_engine.train_kwarg.mean == 0.0122982
    assert test_engine.eval_kwarg.mean == 0.618
