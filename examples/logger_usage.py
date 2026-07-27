"""
Logger Usage Examples

This example demonstrates the comprehensive logger module capabilities
including different field types, training loops, and data export.
"""

import random
import tempfile
from pathlib import Path

# DataFrame export needs the optional 'dataframe' extra; the rest of the
# example runs on a bare `pip install expedantic`.
try:
    import polars  # noqa: F401

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    MaxField,
    MinField,
    SumField,
    ListField,
    CountField,
    StdField,
    MedianField,
)


def basic_logger_example():
    """Basic logger usage with different field types."""

    class BasicLogger(LoggerBase):
        """Demonstrates all field types."""

        # Simple value storage
        iteration: Field[int]
        current_lr: Field[float]

        # Statistical aggregations
        loss: MeanField  # Average
        best_acc: MaxField[float]  # Maximum
        min_loss: MinField[float]  # Minimum
        total_time: SumField[float]  # Sum
        loss_std: StdField  # Standard deviation
        loss_median: MedianField  # Median

        # Collections and counts
        messages: ListField[str]  # List of all values
        error_count: CountField  # Count of log calls

    print("=== Basic Logger Example ===")
    logger = BasicLogger()

    # Simulate some data logging
    for i in range(5):
        logger.iteration.log(i)
        logger.current_lr.log(0.01 * (0.9**i))

        # Log multiple values for aggregation
        loss_val = 1.0 - i * 0.1 + random.gauss(0, 0.05)
        logger.loss.log(loss_val)
        logger.min_loss.log(loss_val)
        logger.loss_std.log(loss_val)
        logger.loss_median.log(loss_val)

        logger.best_acc.log(0.5 + i * 0.1 + random.gauss(0, 0.02))
        logger.total_time.log(0.5 + random.random() * 0.2)

        logger.messages.log(f"Step {i} completed")

        # Count some errors (value doesn't matter for CountField)
        if i % 2 == 0:
            logger.error_count.log("some error")

    # Flush and see results
    entry = logger.flush()
    print("Final entry:", entry)

    # Convert to DataFrame
    if HAS_POLARS:
        df = logger.to_dataframe()
        print("\nDataFrame:")
        print(df)
    else:
        print("\nSkipping DataFrame export: pip install 'expedantic[dataframe]'")

    return logger


def training_logger_example():
    """Realistic ML training loop example."""

    class TrainingLogger(LoggerBase):
        """Logger optimized for ML training."""

        # Training state
        epoch: Field[int]
        iteration: Field[int]
        learning_rate: Field[float]

        # Training metrics (averaged per epoch)
        train_loss: MeanField
        train_acc: MeanField

        # Validation metrics (averaged per epoch)
        val_loss: MeanField
        val_acc: MaxField[float]  # Track best validation accuracy

        # Timing and resources
        epoch_time: SumField[float]
        memory_usage: MaxField[float]

        # Event logging
        events: ListField[str]

    print("\n=== Training Logger Example ===")
    logger = TrainingLogger()

    # Simulate 3 epochs of training
    for epoch in range(3):
        logger.epoch.log(epoch)
        logger.learning_rate.log(0.001 * (0.95**epoch))

        # Training batches
        for batch in range(5):  # 5 batches per epoch
            iteration = epoch * 5 + batch
            logger.iteration.log(iteration)

            # Simulate improving training metrics
            train_loss = 2.0 - epoch * 0.3 - batch * 0.05 + random.gauss(0, 0.1)
            train_acc = 0.3 + epoch * 0.2 + batch * 0.05 + random.gauss(0, 0.02)

            logger.train_loss.log(max(0.1, train_loss))  # Ensure positive loss
            logger.train_acc.log(min(1.0, max(0.0, train_acc)))  # Clamp accuracy

            # Simulate timing
            logger.epoch_time.log(0.5 + random.random() * 0.3)
            logger.memory_usage.log(1000 + random.random() * 200)  # MB

        # Validation phase
        for val_batch in range(2):  # 2 validation batches
            val_loss = 1.8 - epoch * 0.25 + random.gauss(0, 0.05)
            val_acc = 0.4 + epoch * 0.25 + random.gauss(0, 0.03)

            logger.val_loss.log(max(0.1, val_loss))
            logger.val_acc.log(min(1.0, max(0.0, val_acc)))

        # Log epoch events
        logger.events.log(f"Epoch {epoch}: Training completed")
        if epoch == 1:
            logger.events.log("Learning rate decay applied")

        # Flush epoch data
        entry = logger.flush()

        print(f"\nEpoch {epoch} Results:")
        print(f"  Training Loss: {entry['train_loss']:.3f}")
        print(f"  Training Acc: {entry['train_acc']:.3f}")
        print(f"  Val Loss: {entry['val_loss']:.3f}")
        print(f"  Best Val Acc: {entry['val_acc']:.3f}")
        print(f"  Total Time: {entry['epoch_time']:.2f}s")
        print(f"  Max Memory: {entry['memory_usage']:.1f}MB")
        print(f"  Events: {len(entry['events'])}")

    # Final results
    print(f"\nTraining completed! Total epochs logged: {len(logger)}")

    # Export to file
    if HAS_POLARS:
        out = Path(tempfile.gettempdir()) / "training_log.ipc"
        logger.save(out)
        print(f"Results saved to {out}")
    else:
        print("Skipping file export: pip install 'expedantic[dataframe]'")

    return logger


def custom_logger_example():
    """Example of creating domain-specific loggers."""

    class DataProcessingLogger(LoggerBase):
        """Logger for data processing pipelines."""

        # Processing state
        file_name: Field[str]
        records_processed: SumField[int]

        # Performance metrics
        processing_time: SumField[float]
        memory_peak: MaxField[float]

        # Quality metrics
        error_rate: MeanField
        success_rate: MeanField

        # Error tracking
        errors: ListField[str]
        warnings: CountField

    print("\n=== Custom Data Processing Logger ===")
    logger = DataProcessingLogger()

    # Simulate processing multiple files
    files = ["data1.csv", "data2.csv", "data3.csv"]

    for file_name in files:
        logger.file_name.log(file_name)

        # Simulate processing batches within each file
        for batch in range(3):
            # Processing metrics
            # randrange, not randint: numpy's randint upper bound is exclusive.
            records = random.randrange(1000, 5000)
            proc_time = records * 0.001 + random.gauss(0, 0.1)

            logger.records_processed.log(records)
            logger.processing_time.log(max(0.01, proc_time))
            logger.memory_peak.log(500 + random.random() * 300)

            # Quality metrics
            error_rate = max(0, random.gauss(0.02, 0.01))  # ~2% error rate
            logger.error_rate.log(error_rate)
            logger.success_rate.log(1.0 - error_rate)

            # Simulate some errors and warnings
            if random.random() < 0.3:  # 30% chance of error
                logger.errors.log(f"Processing error in {file_name} batch {batch}")

            if random.random() < 0.5:  # 50% chance of warning
                logger.warnings.log("warning")  # Value doesn't matter for CountField

        # Log completion of each file
        entry = logger.flush()

        print(f"\nFile {file_name} processed:")
        print(f"  Records: {entry['records_processed']}")
        print(f"  Time: {entry['processing_time']:.2f}s")
        print(f"  Avg Error Rate: {entry['error_rate']:.3f}")
        print(f"  Avg Success Rate: {entry['success_rate']:.3f}")
        print(f"  Errors: {len(entry['errors']) if 'errors' in entry else 0}")
        print(f"  Warnings: {entry.get('warnings', 0)}")

    return logger


if __name__ == "__main__":
    # Run all examples
    basic_logger = basic_logger_example()
    training_logger = training_logger_example()
    processing_logger = custom_logger_example()

    print("\n" + "=" * 50)
    print("All examples completed successfully!")
    print("Check the generated files and DataFrame outputs above.")
