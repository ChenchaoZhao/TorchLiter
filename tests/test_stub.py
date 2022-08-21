import torchliter


def test_train_stub():

    train = torchliter.stub.Train("loader")

    assert train.dataloader == "loader"

    assert "iteration" in train.__dict__

    assert train.iteration == 0

    print(train)


def test_stub_duplicate():

    assert torchliter.stub.StubBase(kw="kw")(0) is None

    assert len(torchliter.stub.StubBase(kw="kw")(10)) == 10
