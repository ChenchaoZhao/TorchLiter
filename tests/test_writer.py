import os

import liter


def test_writer():

    try:
        os.remove("tmp.csv")
    except FileNotFoundError:
        pass

    with liter.writer.Writer("tmp.csv", ["a", "b"], delimiter=",") as writer:
        writer({"a": 0.1, "b": 0.2})

    with open("tmp.csv", "r") as f:
        string = f.read()
        assert string == "a,b\n0.1,0.2\n"

    with liter.writer.Writer("tmp.csv", ["a", "b"], delimiter=",") as writer:
        writer({"a": 0.1, "b": 0.2})

    with open("tmp.csv", "r") as f:
        string = f.read()
        assert string == "a,b\n0.1,0.2\n0.1,0.2\n"

    os.remove("tmp.csv")
