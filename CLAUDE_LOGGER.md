# CLAUDE_LOGGER.md

This file provides guidance to Claude Code (claude.ai/code) when working with the logger module in the expedantic repository.

## Commands

### Running Logger Examples
```bash
python -c "from expedantic.logger import example_training_loop; example_training_loop()"
```

### Testing
```bash
python -m pytest tests/test_logger.py -v  # Run logger test suite
python -m pytest tests/ -v  # Run all tests including logger
```

### Testing Dependencies
```bash
python -c "import expedantic.logger; print('Logger module loaded successfully')"
```

## Logger Module Architecture

The expedantic package now includes a comprehensive type-safe logging framework designed for machine learning training loops and data collection scenarios. The logger module is located at `src/expedantic/logger.py`.

### Core Architecture

**Field System**: The framework uses a type-safe field system where different field types handle different data aggregation patterns:
- `Field[T]`: Stores the last logged value
- `MeanField`: Computes average of all logged values  
- `MaxField`/`MinField`: Tracks maximum/minimum value across all logs
- `SumField`: Accumulates all logged values
- `ListField`: Collects all logged values in a list (returns copy to prevent mutation)
- `CountField`: Counts occurrences (ignores actual values)
- `StdField`/`MedianField`: Computes statistical measures

**Logger Base Class**: `LoggerBase` uses Python type hints and introspection to automatically instantiate field objects based on class annotations. It provides:
- Automatic schema detection from type annotations
- `flush()` method to compute aggregations and store entries (with automatic timestamping)
- Field reset after flush to prevent data leakage between entries
- `to_dataframe()` method for Polars DataFrame conversion
- `save()` method for IPC format persistence
- Input validation and comprehensive error handling

**Type System**: Extensive use of Python's typing system with:
- Generic type variables for type safety
- Protocol definitions for rich comparison support
- Support for standard Python types (int, float, str, datetime, etc.)

### Dependencies

- **numpy**: For numerical computations and aggregations
- **polars**: For DataFrame operations and data export
- **Standard Library**: typing, inspect, dataclasses, datetime, pathlib, threading

### Usage Pattern

```python
from expedantic.logger import LoggerBase, Field, MeanField, MaxField

class MyLogger(LoggerBase):
    iteration: Field[int]
    loss: MeanField
    accuracy: MaxField[float]

logger = MyLogger()
logger.iteration.log(1)
logger.loss.log(0.5)
logger.accuracy.log(0.95)
entry = logger.flush()  # Computes aggregations and stores entry
df = logger.to_dataframe()  # Export to DataFrame
```

### Example Implementation

The `TrainingLogger` class demonstrates the typical usage pattern for ML training scenarios, with fields for iteration tracking, loss metrics, learning rates, and timing measurements.

### Integration with Expedantic

The logger module is fully integrated into the expedantic package:
- Import with `from expedantic import logger` or `from expedantic.logger import LoggerBase`
- Dependencies are managed through expedantic's pyproject.toml
- Tests are included in the expedantic test suite