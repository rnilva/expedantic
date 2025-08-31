# Multi-Sink Logger Examples

This directory contains examples demonstrating the multi-sink logger functionality in Expedantic.

## Overview

The multi-sink architecture allows you to send logged data to multiple destinations simultaneously when you call `flush()`. This is particularly useful for ML experiments where you want to:

- See real-time output in the console
- Store structured logs in files  
- Track metrics in WandB or MLflow
- Visualize training curves in TensorBoard

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

### Basic Multi-Sink Usage

```python
from expedantic.logger import LoggerBase, Field, MeanField, ConsoleSink, FileSink

class MyLogger(LoggerBase):
    iteration: Field[int] 
    loss: MeanField

# Create logger with multiple sinks
logger = MyLogger(
    name="MyExperiment",
    sinks=[
        ConsoleSink(show_timestamp=True),
        FileSink("experiment.jsonl")
    ]
)

# Log data
logger.iteration.log(1)
logger.loss.log(0.5)
logger.flush()  # Sends to all sinks

# Clean up
logger.close()
```

### Context Manager Usage

```python
with MyLogger(sinks=[console_sink, file_sink]) as logger:
    logger.iteration.log(1)
    logger.loss.log(0.5)
    logger.flush()
# Sinks are automatically closed
```

### Dynamic Sink Management

```python
logger = MyLogger()

# Add sinks dynamically
logger.add_sink(ConsoleSink())
logger.add_sink(FileSink("logs.jsonl"))

# Remove sinks
logger.remove_sink(console_sink)
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

- `multi_sink_example.py` - Basic console + file usage
- `comprehensive_sinks_example.py` - Full ML training scenario
- `tests/test_multi_sink.py` - Complete test coverage

## Optional Dependencies

Some sinks require additional packages:

- **WandBSink**: `pip install wandb`
- **TensorBoardSink**: `pip install tensorboardX` or `pip install torch`