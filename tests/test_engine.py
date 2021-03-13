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


def test_engine():

    trainer = SimpleEngine()

    trainer(liter.stub.Train("dataloader")(2))

    assert trainer.epoch == 2
    assert trainer.iteration == 1000 // 10
    assert trainer.total_iteration == 2000 // 10
