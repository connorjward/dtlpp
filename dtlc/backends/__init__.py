import enum


class Backend(enum.Enum):
    PYTHON = enum.auto()


import dtlc.backends.python  # noqa: F401
