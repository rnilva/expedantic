# Multi-Sink Logger Examples

This directory contains examples demonstrating the multi-sink logger functionality in Expedantic.

## Overview

The multi-sink architecture allows you to send logged data to multiple destinations simultaneously when you call `flush()`. This is particularly useful for ML experiments where you want to:

- See real-time output in the console
- Store structured logs in files  
- Track metrics in WandB or MLflow
- Visualize training curves in TensorBoard

## Configuration Methods

### 1. Decorator-Based Configuration (Recommended)

Use the `@logger_sinks` decorator to configure sinks directly on the class:

```python
from expedantic.logger import LoggerBase, Field, MeanField, logger_sinks, ConsoleSink, FileSink

@logger_sinks([ConsoleSink(), FileSink("training.log")])
class TrainingLogger(LoggerBase):
    epoch: Field[int]
    loss: MeanField

# Logger automatically uses decorated sinks
logger = TrainingLogger()
```

### 2. Explicit Constructor Configuration

Override decorator settings by passing sinks to the constructor:

```python
# Uses decorator sinks
logger = TrainingLogger()

# Overrides with custom sinks  
logger = TrainingLogger(sinks=[FileSink("custom.log")])
```

### 3. Default Behavior

Loggers without decorators or explicit sinks get a ConsoleSink by default:

```python
class SimpleLogger(LoggerBase):
    value: Field[int]

logger = SimpleLogger()  # Automatically gets ConsoleSink()
```

## Available Sinks

### Built-in Sinks

1. **ConsoleSink** - Rich-formatted console output
2. **FileSink** - JSON Lines file storage  
3. **WandBSink** - Weights & Biases integration
4. **TensorBoardSink** - TensorBoard scalar logging

### Creating Custom Sinks

Implement the `SinkProtocol`:

```python
class CustomSink:
    def write(self, data: dict, logger_name: str | None = None) -> None:
        # Handle the logged data
        pass
    
    def close(self) -> None:
        # Clean up resources
        pass
```

## Examples

### Basic Decorator Usage

```python
from expedantic.logger import LoggerBase, Field, MeanField, logger_sinks, ConsoleSink, FileSink

@logger_sinks([ConsoleSink(show_timestamp=True), FileSink("experiment.jsonl")])
class MyLogger(LoggerBase):
    iteration: Field[int] 
    loss: MeanField

# Logger automatically uses decorated sinks
logger = MyLogger()

# Log data
logger.iteration.log(1)
logger.loss.log(0.5)
logger.flush()  # Sends to all decorated sinks

# Clean up
logger.close()
```

### Explicit Constructor Usage

```python
from expedantic.logger import LoggerBase, Field, MeanField, ConsoleSink, FileSink

class MyLogger(LoggerBase):
    iteration: Field[int] 
    loss: MeanField

# Create logger with explicit sinks (bypasses default ConsoleSink)
logger = MyLogger(
    name="MyExperiment",
    sinks=[
        ConsoleSink(show_timestamp=True),
        FileSink("experiment.jsonl")
    ]
)
```

### Context Manager Usage

```python
@logger_sinks([ConsoleSink(), FileSink("training.log")])
class MyLogger(LoggerBase):
    iteration: Field[int]
    loss: MeanField

with MyLogger() as logger:
    logger.iteration.log(1)
    logger.loss.log(0.5)
    logger.flush()
# Sinks are automatically closed
```

### Combined Decorators

```python
from expedantic.logger import logger_sinks, logger_name

@logger_sinks([ConsoleSink(), FileSink("training.log")])
@logger_name("MLExperiment")  # Custom default name
class MyLogger(LoggerBase):
    iteration: Field[int]
    loss: MeanField

logger = MyLogger()  # Uses name "MLExperiment" and decorated sinks
```

### Dynamic Sink Management

```python
logger = MyLogger()

# Add sinks dynamically (adds to decorated sinks)
logger.add_sink(FileSink("additional.log"))

# Remove sinks
logger.remove_sink(some_sink)
```

## Error Handling

Sinks are designed to fail gracefully:

- If a sink fails, it emits a warning but doesn't crash logging
- Other sinks continue to work normally
- Use `logger.close()` to properly clean up resources

## Performance Considerations

- Sinks run synchronously during `flush()`
- File I/O and network calls may impact performance
- Consider batching or async sinks for high-frequency logging

## Integration Examples

See the example files:

- `decorator_sinks_example.py` - Decorator-based configuration examples
- `multi_sink_example.py` - Basic console + file usage
- `comprehensive_sinks_example.py` - Full ML training scenario  
- `tests/test_multi_sink.py` - Complete test coverage
- `tests/test_logger_decorators.py` - Decorator functionality tests

## Optional Dependencies

Some sinks require additional packages:

- **WandBSink**: `pip install wandb`
- **TensorBoardSink**: `pip install tensorboardX` or `pip install torch`