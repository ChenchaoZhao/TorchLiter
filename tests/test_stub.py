import liter


def test_train_stub():

    train = liter.stub.Train("loader")

    assert train.dataloader == "loader"

    assert "iteration" in train.__dict__

    assert train.iteration == 0

    print(train)


def test_stub_duplicate():

    assert liter.stub.StubBase(kw="kw")(0) is None

    assert len(liter.stub.StubBase(kw="kw")(10)) == 10
