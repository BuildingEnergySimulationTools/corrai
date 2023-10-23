from enum import Enum


class Parameter(Enum):
    """
    NAME: the model path to the parameter
    INTERVAL: a tuple indicating parameter bounds
    INIT_VALUE: the parameter initial value
    TYPE: Integer, Real, Choice, Binary
    """

    NAME = "NAME"
    INTERVAL = "INTERVAL"
    INIT_VALUE = "INIT_VALUE"
    TYPE = "TYPE"
