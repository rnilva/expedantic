"""Example demonstrating decorator-based sink configuration.

This example shows the clean decorator syntax for configuring logger sinks
directly on the class definition, making the configuration more declarative
and discoverable.
"""

import tempfile
from pathlib import Path

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    MaxField,
    SinkProtocol,
    ConsoleSink,
    FileSink,
    logger_sinks,
    logger_name,
)


# Example 1: Basic decorator usage
@logger_sinks([ConsoleSink(show_timestamp=True)])
class SimpleLogger(LoggerBase):
    """A simple logger with console output only."""

    step: Field[int]
    value: MeanField


# Example 2: Multiple sinks with decorator
@logger_sinks([ConsoleSink(show_timestamp=True), FileSink("training.jsonl", mode="w")])
@logger_name("MLTraining")
class TrainingLogger(LoggerBase):
    """A training logger with both console and file output."""

    epoch: Field[int]
    loss: MeanField
    accuracy: MaxField[float]
    learning_rate: Field[float]


# Example 3: Production logger with comprehensive sinks
@logger_sinks(
    [
        ConsoleSink(show_timestamp=True, show_logger_name=True),
        FileSink("production.jsonl", mode="a"),  # Append mode for production
    ]
)
@logger_name("ProductionML")
class ProductionLogger(LoggerBase):
    """Production logger with multiple metrics."""

    iteration: Field[int]
    epoch: Field[int]
    train_loss: MeanField
    val_loss: MeanField
    accuracy: MaxField[float]
    f1_score: MaxField[float]
    memory_usage: MaxField[float]


def demonstrate_basic_usage():
    """Show basic decorator usage."""
    print("🔥 Example 1: Basic Decorator Usage")
    print("=" * 50)

    # Logger automatically uses ConsoleSink from decorator
    logger = SimpleLogger()
    print(f"Logger name: {logger.name}")
    print(f"Number of sinks: {len(logger.sinks)}")
    print(f"Sink types: {[type(sink).__name__ for sink in logger.sinks]}")

    # Log some data
    for i in range(3):
        logger.step.log(i)
        logger.value.log(0.5 + i * 0.1)
        logger.flush()

    print()


def demonstrate_multiple_sinks():
    """Show multiple sinks with decorator."""
    print("🔥 Example 2: Multiple Sinks with Decorators")
    print("=" * 50)

    # Create temporary file for demo
    temp_file = Path(tempfile.mktemp(suffix=".jsonl"))

    try:
        # Use decorator-defined sinks
        logger = TrainingLogger()
        # Note: decorator defined FileSink with hardcoded path, but we'll override
        logger = TrainingLogger(
            sinks=[ConsoleSink(show_timestamp=True), FileSink(temp_file, mode="w")]
        )

        print(f"Logger name: {logger.name}")  # Uses @logger_name
        print(f"Number of sinks: {len(logger.sinks)}")
        print(f"Sink types: {[type(sink).__name__ for sink in logger.sinks]}")

        # Simulate training
        for epoch in range(2):
            logger.epoch.log(epoch)
            logger.learning_rate.log(0.001 * (0.9**epoch))

            for batch in range(3):
                logger.loss.log(0.5 - epoch * 0.1 - batch * 0.02)
                logger.accuracy.log(0.7 + epoch * 0.15 + batch * 0.05)

            logger.flush()

        print(f"\n📄 File contents ({temp_file}):")
        with open(temp_file) as f:
            for i, line in enumerate(f, 1):
                print(f"  {i}: {line.strip()}")

    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()

    print()


def demonstrate_override_behavior():
    """Show how explicit sinks override decorator sinks."""
    print("🔥 Example 3: Override Decorator Sinks")
    print("=" * 50)

    # Default behavior - uses decorator sinks
    default_logger = SimpleLogger()
    print(f"Default sinks: {[type(s).__name__ for s in default_logger.sinks]}")

    # Override with explicit sinks
    custom_sinks: list[SinkProtocol] = [ConsoleSink(show_timestamp=False)]
    override_logger = SimpleLogger(sinks=custom_sinks)
    print(f"Override sinks: {[type(s).__name__ for s in override_logger.sinks]}")

    # Test both loggers
    for name, logger in [("Default", default_logger), ("Override", override_logger)]:
        print(f"\n{name} logger output:")
        logger.step.log(42)
        logger.value.log(3.14)
        logger.flush()

    print()


def demonstrate_no_decorator():
    """Show default behavior without decorators."""
    print("🔥 Example 4: No Decorator (Default ConsoleSink)")
    print("=" * 50)

    class PlainLogger(LoggerBase):
        """Logger without any decorators."""

        message: Field[str]
        count: Field[int]

    # Gets default ConsoleSink automatically
    logger = PlainLogger()
    print(f"Logger name: {logger.name}")  # Uses class name
    print(f"Default sinks: {[type(s).__name__ for s in logger.sinks]}")

    logger.message.log("Hello from plain logger!")
    logger.count.log(100)
    logger.flush()

    print()


def main():
    """Run all demonstrations."""
    print("🚀 Logger Decorator Examples")
    print("=" * 60)
    print()

    demonstrate_basic_usage()
    demonstrate_multiple_sinks()
    demonstrate_override_behavior()
    demonstrate_no_decorator()

    print("✅ All examples completed!")


if __name__ == "__main__":
    main()
