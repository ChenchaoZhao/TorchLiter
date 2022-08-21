import os

import torchliter


def test_csv_writer():

    try:
        os.remove("tmp.csv")
    except FileNotFoundError:
        pass

    with torchliter.writer.CSVWriter("tmp.csv", ["a", "b"], delimiter=",") as writer:
        print(writer)
        writer({"a": 0.1, "b": 0.2})

    with open("tmp.csv", "r") as f:
        string = f.read()
        assert string == "a,b\n0.1,0.2\n"

    with torchliter.writer.CSVWriter("tmp.csv", ["a", "b"], delimiter=",") as writer:
        writer({"a": 0.1, "b": 0.2})

    with open("tmp.csv", "r") as f:
        string = f.read()
        assert string == "a,b\n0.1,0.2\n0.1,0.2\n"

    os.remove("tmp.csv")
