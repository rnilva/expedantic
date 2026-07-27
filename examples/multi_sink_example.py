"""Example demonstrating multi-sink logger functionality.

This example shows how to set up a logger with multiple sinks including
console output, file logging, and placeholder WandB/TensorBoard sinks.
"""

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    MaxField,
    ConsoleSink,
    FileSink,
)
import tempfile
from pathlib import Path


class ExperimentLogger(LoggerBase):
    """Logger for machine learning experiments."""

    epoch: Field[int]
    loss: MeanField
    accuracy: MaxField[float]
    learning_rate: Field[float]


def main():
    # Create temporary file for file sink
    temp_dir = Path(tempfile.mkdtemp())
    log_file = temp_dir / "experiment.jsonl"

    # Set up multiple sinks
    console_sink = ConsoleSink(show_timestamp=True, show_logger_name=True)
    file_sink = FileSink(log_file, mode="w")

    # Create logger with multiple sinks
    logger = ExperimentLogger(name="TrainingRun", sinks=[console_sink, file_sink])

    print("🚀 Starting training with multi-sink logging...")
    print(f"📄 Log file: {log_file}")
    print()

    # Simulate training loop
    for epoch in range(3):
        logger.epoch.log(epoch)
        logger.learning_rate.log(0.001 * (0.9**epoch))

        # Simulate multiple batches per epoch
        for batch in range(5):
            logger.loss.log(0.5 - epoch * 0.1 + (batch * 0.01))
            logger.accuracy.log(0.7 + epoch * 0.15 + (batch * 0.02))

        # Flush sends data to all sinks
        entry = logger.flush()

        print()  # Add spacing between epochs

    # Close all sinks properly
    logger.close()

    print("✅ Training complete!")
    print(f"📊 Generated {len(logger)} log entries")

    # Show file contents
    print(f"\n📄 File contents ({log_file}):")
    with open(log_file) as f:
        for line_num, line in enumerate(f, 1):
            print(f"  {line_num}: {line.strip()}")

    # Cleanup
    log_file.unlink()
    temp_dir.rmdir()


if __name__ == "__main__":
    main()
