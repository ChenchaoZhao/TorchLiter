import csv
import os

from . import REPR_INDENT

__all__ = ["CSVWriter"]


class CSVWriter:
    """CSV writer."""

    def __init__(self, path, columns, delimiter=","):
        self.path = path
        self.path_exists = os.path.exists(path)
        self.delimiter = delimiter
        if self.path_exists:
            with open(path, "r") as f:
                header = next(csv.reader(f, delimiter=self.delimiter))
            if set(header) == set(columns):
                self.columns = header
                self.write_header = False
            else:
                raise ValueError(
                    "Header in file is inconsistent with columns: "
                    f"header: {header}; columns {columns}"
                )
        else:
            self.columns = columns
            self.write_header = True

        self.file = None
        self.writer = None

    def open(self):

        mode = "a+" if self.path_exists else "w"
        self.file = open(self.path, mode, buffering=1)
        self.writer = csv.DictWriter(
            f=self.file, fieldnames=self.columns, delimiter=self.delimiter
        )
        if self.write_header:
            self.writer.writeheader()

    def close(self):
        self.file.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write_row(self, row_dict):
        self.writer.writerow(row_dict)

    def __call__(self, row_dict):
        self.write_row(row_dict)

    def __repr__(self):
        out = []
        out.append(self.__class__.__name__)
        out.append(" " * REPR_INDENT + f"filepath: {self.path}")
        out.append(" " * REPR_INDENT + f"columns: {self.columns}")
        return "\n".join(out)
