"""Comprehensive test suite for the logger module."""

import statistics

import pytest
from datetime import datetime
from io import BytesIO

from expedantic.logger import (
    Field,
    MeanField,
    MaxField,
    MinField,
    SumField,
    ListField,
    CountField,
    StdField,
    MedianField,
    LoggerBase,
    TrainingLogger,
)


class TestFieldTypes:
    """Test all field types individually."""

    def test_field_basic_functionality(self):
        """Test Field stores and returns the last value."""
        field = Field[int]()

        # Initial state
        assert field.value is None

        # Log values
        field.log(10)
        assert field.value == 10

        field.log(20)
        assert field.value == 20  # Should overwrite

    def test_field_none_validation(self):
        """Test Field rejects None values."""
        field = Field[int]()

        with pytest.raises(ValueError, match="Cannot log None value"):
            field.log(None)

    def test_mean_field(self):
        """Test MeanField computes correct averages."""
        field = MeanField()

        # Empty case
        assert field.value is None

        # Single value
        field.log(10.0)
        assert field.value == 10.0

        # Multiple values
        field.log(20.0)
        field.log(30.0)
        assert field.value == 20.0  # (10 + 20 + 30) / 3

    def test_mean_field_error_handling(self):
        """Test MeanField handles non-numeric values gracefully."""
        field = MeanField()
        field.log("not a number")

        with pytest.raises(ValueError, match="Cannot compute mean"):
            _ = field.value

    def test_max_field(self):
        """Test MaxField tracks maximum values."""
        field = MaxField[float]()

        # Empty case
        assert field.value is None

        # Add values
        field.log(10.0)
        assert field.value == 10.0

        field.log(5.0)
        assert field.value == 10.0  # Max is still 10

        field.log(15.0)
        assert field.value == 15.0  # New max

    def test_min_field(self):
        """Test MinField tracks minimum values."""
        field = MinField[float]()

        # Empty case
        assert field.value is None

        # Add values
        field.log(10.0)
        assert field.value == 10.0

        field.log(15.0)
        assert field.value == 10.0  # Min is still 10

        field.log(5.0)
        assert field.value == 5.0  # New min

    def test_sum_field(self):
        """Test SumField accumulates values correctly."""
        field = SumField[int]()

        # Empty case - should return 0
        assert field.value == 0

        field.log(10)
        assert field.value == 10

        field.log(20)
        assert field.value == 30

        field.log(-5)
        assert field.value == 25

    def test_list_field(self):
        """Test ListField collects all values."""
        field = ListField[str]()

        # Empty case - should return None
        assert field.value is None

        field.log("first")
        assert field.value == ["first"]

        field.log("second")
        assert field.value == ["first", "second"]

    def test_count_field(self):
        """Test CountField counts log calls."""
        field = CountField()

        # Initial count
        assert field.value == 0

        # Count increments regardless of value
        field.log("anything")
        assert field.value == 1

        field.log(None)  # CountField allows None
        assert field.value == 2

        field.log(42)
        assert field.value == 3

    def test_std_field(self):
        """Test StdField computes standard deviation."""
        field = StdField()

        # Empty case
        assert field.value is None

        # Single value (std should be 0)
        field.log(10.0)
        assert field.value == 0.0

        # Multiple values
        field.log(12.0)
        field.log(8.0)
        # std of [10, 12, 8] ≈ 1.63
        assert abs(field.value - statistics.pstdev([10.0, 12.0, 8.0])) < 1e-10

    def test_median_field(self):
        """Test MedianField computes median correctly."""
        field = MedianField()

        # Empty case
        assert field.value is None

        # Single value
        field.log(10.0)
        assert field.value == 10.0

        # Odd number of values
        field.log(20.0)
        field.log(5.0)
        assert field.value == 10.0  # median of [10, 20, 5] = 10

        # Even number of values
        field.log(15.0)
        assert field.value == 12.5  # median of [10, 20, 5, 15] = 12.5


class TestLoggerBase:
    """Test the LoggerBase functionality."""

    def test_simple_logger_creation(self):
        """Test creating a simple logger with type annotations."""

        class SimpleLogger(LoggerBase):
            counter: Field[int]
            average: MeanField

        logger = SimpleLogger()

        # Fields should be automatically instantiated
        assert hasattr(logger, "counter")
        assert hasattr(logger, "average")
        assert isinstance(logger.counter, Field)
        assert isinstance(logger.average, MeanField)

    def test_logger_flush_functionality(self):
        """Test the flush mechanism."""

        class TestLogger(LoggerBase):
            value: Field[int]
            total: SumField[int]

        logger = TestLogger()

        # Log some values
        logger.value.log(42)
        logger.total.log(10)
        logger.total.log(20)

        # Flush and check results
        entry = logger.flush()

        assert entry["value"] == 42
        assert entry["total"] == 30
        assert "_timestamp" in entry
        assert isinstance(entry["_timestamp"], datetime)

        # Check data is stored
        assert len(logger) == 1
        assert logger.data[0] == entry

    def test_logger_multiple_flushes(self):
        """Test multiple flush operations."""

        class TestLogger(LoggerBase):
            iteration: Field[int]
            score: MeanField

        logger = TestLogger()

        # First epoch
        logger.iteration.log(1)
        logger.score.log(0.8)
        logger.score.log(0.9)
        entry1 = logger.flush()

        # Second epoch
        logger.iteration.log(2)
        logger.score.log(0.85)
        logger.score.log(0.95)
        entry2 = logger.flush()

        assert len(logger) == 2
        assert entry1["iteration"] == 1
        assert abs(entry1["score"] - 0.85) < 1e-10  # (0.8 + 0.9) / 2
        assert entry2["iteration"] == 2
        assert abs(entry2["score"] - 0.9) < 1e-10  # (0.85 + 0.95) / 2

    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        pytest.importorskip("polars", reason="requires the 'dataframe' extra")

        class TestLogger(LoggerBase):
            step: Field[int]
            value: Field[float]

        logger = TestLogger()

        # Add some data
        logger.step.log(1)
        logger.value.log(1.5)
        logger.flush()

        logger.step.log(2)
        logger.value.log(2.5)
        logger.flush()

        df = logger.to_dataframe()

        assert len(df) == 2
        assert "step" in df.columns
        assert "value" in df.columns
        assert "_timestamp" in df.columns

        # Check data values
        steps = df["step"].to_list()
        values = df["value"].to_list()
        assert steps == [1, 2]
        assert values == [1.5, 2.5]

    def test_save_functionality(self):
        """Test saving to BytesIO buffer."""
        pytest.importorskip("polars", reason="requires the 'dataframe' extra")

        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger()

        logger.value.log(100)
        logger.flush()

        # Save to buffer
        buffer = BytesIO()
        logger.save(buffer)

        # Buffer should contain data
        assert len(buffer.getvalue()) > 0


class TestTrainingLogger:
    """Test the example TrainingLogger implementation."""

    def test_training_logger_schema(self):
        """Test TrainingLogger has correct field types."""
        logger = TrainingLogger()

        # Check all expected fields exist and have correct types
        assert isinstance(logger.iteration, Field)
        assert isinstance(logger.epoch, Field)
        assert isinstance(logger.loss, MeanField)
        assert isinstance(logger.val_loss, MeanField)
        assert isinstance(logger.learning_rate, Field)
        assert isinstance(logger.messages, ListField)
        assert isinstance(logger.accuracy, MaxField)
        assert isinstance(logger.batch_time, SumField)

    def test_training_logger_realistic_usage(self):
        """Test realistic training scenario."""
        logger = TrainingLogger()

        # Simulate training epoch
        logger.epoch.log(1)
        logger.learning_rate.log(0.001)

        # Simulate multiple batches
        for batch in range(3):
            logger.iteration.log(batch)
            logger.loss.log(0.5 - batch * 0.1)  # Decreasing loss
            logger.accuracy.log(0.7 + batch * 0.1)  # Increasing accuracy
            logger.batch_time.log(0.1)

        # Add validation data
        logger.val_loss.log(0.4)
        logger.val_loss.log(0.35)
        logger.messages.log("Training complete")
        logger.messages.log("Validation complete")

        # Flush and check results
        entry = logger.flush()

        assert entry["epoch"] == 1
        assert entry["iteration"] == 2  # Last iteration
        assert entry["learning_rate"] == 0.001
        assert (
            abs(entry["loss"] - 0.4) < 1e-10
        )  # Mean of [0.5, 0.4, 0.3] = (0.5+0.4+0.3)/3 = 1.2/3 = 0.4
        assert abs(entry["val_loss"] - 0.375) < 1e-10  # Mean of [0.4, 0.35]
        assert abs(entry["accuracy"] - 0.9) < 1e-10  # Max of [0.7, 0.8, 0.9]
        assert abs(entry["batch_time"] - 0.3) < 1e-10  # Sum of [0.1, 0.1, 0.1]
        # Messages should be captured in entry before reset
        assert "messages" in entry
        assert len(entry["messages"]) == 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_field_none_validation(self):
        """Test that fields properly reject None values."""
        field = Field[str]()
        mean_field = MeanField()

        with pytest.raises(ValueError):
            field.log(None)

        with pytest.raises(ValueError):
            mean_field.log(None)

    def test_comparison_field_error_handling(self):
        """Test error handling for incomparable types."""
        max_field = MaxField[object]()
        min_field = MinField[object]()

        # These should work
        max_field.log(5)
        min_field.log(5)

        # Add incomparable type
        max_field.log("string")
        min_field.log("string")

        # Should raise TypeError when trying to compare
        with pytest.raises(TypeError):
            _ = max_field.value

        with pytest.raises(TypeError):
            _ = min_field.value

    def test_numeric_field_error_handling(self):
        """Test error handling for non-numeric operations."""
        std_field = StdField()
        median_field = MedianField()

        std_field.log("not a number")
        median_field.log("not a number")

        with pytest.raises(ValueError, match="Cannot compute standard deviation"):
            _ = std_field.value

        with pytest.raises(ValueError, match="Cannot compute median"):
            _ = median_field.value


if __name__ == "__main__":
    pytest.main([__file__])
