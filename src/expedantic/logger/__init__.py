"""Multi-sink logging framework for Expedantic.

This module provides a type-safe logging system with field-based aggregation
and support for multiple output sinks (console, file, WandB, TensorBoard, etc.).

The logger system is built around three main concepts:

1. **Fields**: Define how individual metrics are aggregated (mean, max, sum, etc.)
2. **LoggerBase**: The main logger class that manages fields and sinks
3. **Sinks**: Output destinations that receive flushed data

Basic usage with decorators (recommended):

    from expedantic.logger import LoggerBase, Field, MeanField, logger_sinks, ConsoleSink

    @logger_sinks([ConsoleSink()])
    class MyLogger(LoggerBase):
        iteration: Field[int]
        loss: MeanField

    logger = MyLogger()  # Uses decorated sinks
    logger.iteration.log(1)
    logger.loss.log(0.5)
    logger.flush()  # Sends to all sinks

Alternative explicit usage:

    class MyLogger(LoggerBase):
        iteration: Field[int]
        loss: MeanField

    logger = MyLogger(sinks=[ConsoleSink()])  # Explicit sinks
    logger = MyLogger()  # Gets default ConsoleSink

For more complex scenarios, see examples in the examples/ directory.
"""

# Import all field types
from .fields import (
    FieldBase,
    ReducibleFieldBase,
    Field,
    MeanField,
    MaxField,
    MinField,
    StdField,
    MedianField,
    SumField,
    ListField,
    CountField,
    SupportedTypes,
)

# Import sink protocol and implementations
from .sinks import (
    SinkProtocol,
    ConsoleSink,
    FileSink,
    WandBSink,
    TensorBoardSink,
)

# Import main logger class
from .base import LoggerBase

# Import decorators
from .decorators import logger_sinks, logger_name

# Import example loggers
from .examples import TrainingLogger

# Define what gets imported with "from expedantic.logger import *"
__all__ = [
    # Field types
    "FieldBase",
    "ReducibleFieldBase",
    "Field",
    "MeanField",
    "MaxField",
    "MinField",
    "StdField",
    "MedianField",
    "SumField",
    "ListField",
    "CountField",
    "SupportedTypes",
    # Sinks
    "SinkProtocol",
    "ConsoleSink",
    "FileSink",
    "WandBSink",
    "TensorBoardSink",
    # Logger base
    "LoggerBase",
    # Decorators
    "logger_sinks",
    "logger_name",
    # Examples
    "TrainingLogger",
]
