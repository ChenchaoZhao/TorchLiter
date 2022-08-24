import torchliter


def test_train_stub():

    s = torchliter.stub.Train("loader")

    assert s.dataloader == "loader"

    assert "iteration" in s.__dict__

    assert s.iteration == 0

    print(s)


def test_evaluate_stub():

    s = torchliter.stub.Evaluate("loader")

    assert s.dataloader == "loader"

    print(s)


def test_lambda_stub():

    s = torchliter.stub.Lambda("_method_name&*")

    assert "action" in s.__dict__
    assert s.action == "_method_name__"

    print(s)


def test_stub_duplicate():

    assert torchliter.stub.StubBase(kw="kw")(0) is None

    assert len(torchliter.stub.StubBase(kw="kw")(10)) == 10
