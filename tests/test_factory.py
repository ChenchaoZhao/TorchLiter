import torchliter.engine.factory as factory


def test_fine_outputs():
    def _gen():
        yield "var1", 1
        yield "var2", 2
        yield "var3", 3

    assert factory._find_outputs(_gen) == ["var1", "var2", "var3"]
