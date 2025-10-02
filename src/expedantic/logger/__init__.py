"""Multi-sink logging framework for Expedantic.

This module provides a type-safe logging system with field-based aggregation
and support for multiple output sinks (console, file, WandB, TensorBoard, etc.).

The logger system is built around three main concepts:

1. **Fields**: Define how individual metrics are aggregated (mean, max, sum, etc.)
2. **LoggerBase**: The main logger class that manages fields and sinks
3. **Sinks**: Output destinations that receive flushed data

## Configuration Approaches

### 1. Class Attributes (Great for IDE autocompletion)

    from expedantic.logger import LoggerBase, Field, MeanField, ConsoleSink, FileSink

    class MyLogger(LoggerBase):
        _sinks = [ConsoleSink(), FileSink("training.log")]
        _name = "TrainingLogger"

        iteration: Field[int]
        loss: MeanField

    logger = MyLogger()  # Uses class attribute sinks and name

### 2. Decorators (Recommended for complex setups)

    from expedantic.logger import LoggerBase, Field, MeanField, logger_sinks, logger_name

    @logger_sinks([ConsoleSink(), FileSink("training.log")])
    @logger_name("TrainingLogger")
    class MyLogger(LoggerBase):
        iteration: Field[int]
        loss: MeanField

    logger = MyLogger()  # Uses decorated sinks and name

### 3. Runtime Configuration (Flexible)

    class MyLogger(LoggerBase):
        iteration: Field[int]
        loss: MeanField

    logger = MyLogger(name="Custom", sinks=[ConsoleSink()])  # Explicit parameters
    logger = MyLogger()  # Gets default ConsoleSink and class name

## Precedence Order

Configuration follows this precedence (highest to lowest):
1. Constructor parameters (runtime)
2. Class attributes (`_sinks`, `_name`)
3. Decorators (`@logger_sinks`, `@logger_name`)
4. Defaults (ConsoleSink, class name)

## Basic Usage

    logger = MyLogger()
    logger.iteration.log(1)
    logger.loss.log(0.5)
    logger.flush()  # Sends to all configured sinks

For more examples, see the examples/ directory.
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
