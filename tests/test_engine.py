import torch

import liter


class SimpleEngine(liter.engine.EngineBase):
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

    trainer(liter.stub.Train("dataloader")(2))

    assert trainer.epoch == 2
    assert trainer.iteration == 1000 // 10
    assert trainer.total_iteration == 2000 // 10

    state_dict = trainer.state_dict()

    trainer.load_state_dict(state_dict)


def test_automated():

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def classification(engine, batch):

        engine.train()

        x, y = batch
        lgs = engine.model(x)
        loss = F.cross_entropy(lgs, y)

        yield "loss", loss.item()

        acc = (lgs.max(-1).indices == y).float().mean()

        yield "acc", acc.item()

    eng = liter.engine.Automated.from_forward(classification, 100)

    print(eng)

    assert isinstance(eng, liter.engine.Automated)
    assert hasattr(eng, "buffer_registry")
    assert "loss" in eng.buffer_registry
    assert "acc" in eng.buffer_registry
    assert isinstance(eng.loss, liter.engine.buffer.BufferBase)
    assert isinstance(eng.acc, liter.engine.buffer.BufferBase)

    eng = liter.engine.Automated.from_forward(classification)
    eng.attach(model=nn.Linear(2, 2))
    assert "model" in eng.model_registry

    eng.per_batch((torch.randn(3, 2), torch.tensor([1, 0, 1])))

    assert eng.loss._count == 1
    assert eng.acc._count == 1
