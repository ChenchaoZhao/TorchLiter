import torchliter.engine.utils as utils


def test_find_outputs():
    def _gen():
        yield "var1", 1
        yield "var2", 2
        yield "var3", 3

    assert utils._find_output_names(_gen) == ["var1", "var2", "var3"]

    def _gen():
        yield "var1,2", 1
        yield "var2,3", 2
        yield "var3,5", 3

    assert utils._find_output_names(_gen) == ["var1,2", "var2,3", "var3,5"]
