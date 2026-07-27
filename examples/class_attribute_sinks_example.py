"""Example demonstrating class attribute-based sink configuration.

This example shows how to configure logger sinks using class attributes,
which provides excellent IDE autocompletion support and a clean interface.
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
    logger_sinks,  # For comparison
)


# Example 1: Basic class attribute usage
class SimpleLogger(LoggerBase):
    """A simple logger with class attribute configuration."""

    _sinks = [ConsoleSink(show_timestamp=True)]
    _name = "SimpleLog"

    step: Field[int]
    value: MeanField


# Example 2: Multiple sinks with class attributes
class TrainingLogger(LoggerBase):
    """A training logger with multiple sinks configured via class attributes."""

    _sinks = [
        ConsoleSink(show_timestamp=True, show_logger_name=True),
        FileSink("training.jsonl", mode="w"),
    ]
    _name = "MLTraining"

    epoch: Field[int]
    loss: MeanField
    accuracy: MaxField[float]
    learning_rate: Field[float]


# Example 3: Production logger
class ProductionLogger(LoggerBase):
    """Production logger with comprehensive class attribute configuration."""

    _sinks = [
        ConsoleSink(show_timestamp=True, show_logger_name=True),
        FileSink("production.jsonl", mode="a"),  # Append mode for production
    ]
    _name = "ProductionML"

    iteration: Field[int]
    epoch: Field[int]
    train_loss: MeanField
    val_loss: MeanField
    accuracy: MaxField[float]
    f1_score: MaxField[float]
    memory_usage: MaxField[float]


# Example 4: IDE autocompletion demonstration
class AutocompleteLogger(LoggerBase):
    """Logger showing IDE autocompletion benefits."""

    # IDE will provide autocompletion for _sinks and _name
    # Users can see type hints and available attributes
    _sinks = [
        ConsoleSink(),
        # FileSink("autocomplete.log"),  # IDE suggests available sinks
    ]
    _name = "AutoComplete"

    metric: Field[float]


# Example 5: Comparison with decorator approach
@logger_sinks([ConsoleSink()])
class DecoratorLogger(LoggerBase):
    """Decorator-based logger for comparison."""

    metric: Field[float]


def demonstrate_basic_usage():
    """Show basic class attribute usage."""
    print("🔥 Example 1: Class Attribute Usage")
    print("=" * 50)

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
    """Show multiple sinks with class attributes."""
    print("🔥 Example 2: Multiple Sinks via Class Attributes")
    print("=" * 50)

    # Create temporary file for demo
    temp_file = Path(tempfile.mktemp(suffix=".jsonl"))

    try:
        # Override the hardcoded file with temp file
        logger = TrainingLogger(
            sinks=[ConsoleSink(show_timestamp=True), FileSink(temp_file, mode="w")]
        )

        print(f"Logger name: {logger.name}")
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


def demonstrate_precedence():
    """Show precedence of configuration methods."""
    print("🔥 Example 3: Configuration Precedence")
    print("=" * 50)

    # Default behavior - uses class attributes
    default_logger = SimpleLogger()
    print(f"Default logger name: {default_logger.name}")  # Uses _name
    print(
        f"Default sinks: {[type(s).__name__ for s in default_logger.sinks]}"
    )  # Uses _sinks

    # Override with explicit parameters
    custom_sinks: list[SinkProtocol] = [ConsoleSink(show_timestamp=False)]
    override_logger = SimpleLogger(name="OverriddenName", sinks=custom_sinks)
    print(f"Override logger name: {override_logger.name}")  # Explicit wins
    print(
        f"Override sinks: {[type(s).__name__ for s in override_logger.sinks]}"
    )  # Explicit wins

    print()


def demonstrate_inheritance():
    """Show inheritance behavior with class attributes."""
    print("🔥 Example 4: Inheritance with Class Attributes")
    print("=" * 50)

    class BaseLogger(LoggerBase):
        _sinks = [ConsoleSink()]
        _name = "BaseLogger"
        value: Field[int]

    class ChildLogger(BaseLogger):
        _sinks = [ConsoleSink(show_timestamp=True)]
        _name = "ChildLogger"
        extra_field: Field[str]

    base_logger = BaseLogger()
    child_logger = ChildLogger()

    print(f"Base logger: {base_logger.name}")
    print(f"Child logger: {child_logger.name}")
    print(f"Child overrides base attributes")

    print()


def demonstrate_ide_benefits():
    """Show IDE autocompletion benefits."""
    print("🔥 Example 5: IDE Autocompletion Benefits")
    print("=" * 50)

    print("With class attributes:")
    print("- IDE provides autocompletion for _sinks and _name")
    print("- Type hints show list[SinkProtocol] and str")
    print("- Easy to discover available configuration options")
    print("- No magic - just regular Python class attributes")

    # Both approaches work
    class_logger = AutocompleteLogger()
    decorator_logger = DecoratorLogger()

    print(f"\nClass attribute logger: {class_logger.name}")
    print(f"Decorator logger: {decorator_logger.name}")

    print()


def demonstrate_mixed_approaches():
    """Show that different approaches can coexist."""
    print("🔥 Example 6: Mixed Configuration Approaches")
    print("=" * 50)

    # All these approaches work together
    class_logger = SimpleLogger()  # Uses class attributes
    decorator_logger = DecoratorLogger()  # Uses decorators
    plain_logger = LoggerBase()  # Uses defaults
    runtime_logger = LoggerBase(name="Runtime", sinks=[ConsoleSink()])  # Runtime config

    loggers = [
        ("Class Attributes", class_logger),
        ("Decorator", decorator_logger),
        ("Default", plain_logger),
        ("Runtime", runtime_logger),
    ]

    for name, logger in loggers:
        print(f"{name:15}: {logger.name} ({len(logger.sinks)} sinks)")

    print("\nAll approaches are valid and can be used as needed!")
    print()


def main():
    """Run all demonstrations."""
    print("🚀 Logger Class Attribute Examples")
    print("=" * 60)
    print()

    demonstrate_basic_usage()
    demonstrate_multiple_sinks()
    demonstrate_precedence()
    demonstrate_inheritance()
    demonstrate_ide_benefits()
    demonstrate_mixed_approaches()

    print("✅ All examples completed!")


if __name__ == "__main__":
    main()
