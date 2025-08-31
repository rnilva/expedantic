"""Tests for multi-sink logger functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    ConsoleSink,
    FileSink,
    WandBSink,
    TensorBoardSink,
)


class TestLogger(LoggerBase):
    """Simple test logger."""

    step: Field[int]
    value: MeanField


class TestSinks:
    """Test sink implementations."""

    def test_console_sink(self):
        """Test console sink output."""
        with patch("rich.console.Console") as mock_console_cls:
            mock_console = Mock()
            mock_console_cls.return_value = mock_console

            sink = ConsoleSink(show_timestamp=True, show_logger_name=True)

            # Test data with various types
            data = {
                "step": 1,
                "value": 0.5,
                "_timestamp": None,  # Will be set to real datetime in practice
            }

            sink.write(data, "TestLogger")

            # Verify console.print was called
            mock_console.print.assert_called_once()

    def test_file_sink(self):
        """Test file sink JSON output."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
            temp_path = Path(f.name)

        try:
            sink = FileSink(temp_path, mode="w")

            # Test data
            data = {"step": 1, "value": 0.75, "name": "test"}

            sink.write(data, "TestLogger")
            sink.close()

            # Verify file contents
            with open(temp_path, "r") as f:
                line = f.readline().strip()
                parsed = json.loads(line)

                assert parsed["_logger"] == "TestLogger"
                assert parsed["step"] == 1
                assert parsed["value"] == 0.75
                assert parsed["name"] == "test"

        finally:
            temp_path.unlink()

    def test_wandb_sink_no_wandb(self):
        """Test WandB sink when wandb is not available."""
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError, match="wandb package is required"):
                WandBSink()

    def test_wandb_sink_with_mock(self):
        """Test WandB sink with mocked wandb."""
        mock_wandb = Mock()
        mock_wandb.run = Mock()

        with patch.dict("sys.modules", {"wandb": mock_wandb}):
            sink = WandBSink(step_field="iteration", prefix="exp_")

            data = {"iteration": 10, "loss": 0.5, "accuracy": 0.95}

            sink.write(data, "TestLogger")

            # Verify wandb.log was called with correct data
            expected_data = {"exp_iteration": 10, "exp_loss": 0.5, "exp_accuracy": 0.95}
            mock_wandb.log.assert_called_once_with(expected_data, step=10)

    def test_tensorboard_sink_no_tensorboard(self):
        """Test TensorBoard sink when tensorboardX/torch is not available."""
        with patch.dict("sys.modules", {"tensorboardX": None, "torch": None}):
            with pytest.raises(ImportError, match="tensorboardX or torch is required"):
                TensorBoardSink("/tmp/logs")

    def test_tensorboard_sink_with_mock(self):
        """Test TensorBoard sink with mocked tensorboardX."""
        mock_writer = Mock()
        mock_summary_writer = Mock(return_value=mock_writer)

        with patch.dict(
            "sys.modules", {"tensorboardX": Mock(SummaryWriter=mock_summary_writer)}
        ):
            sink = TensorBoardSink("/tmp/logs", step_field="step")

            data = {
                "step": 5,
                "loss": 0.3,
                "accuracy": 0.8,
                "message": "should be ignored",  # Non-numeric
            }

            sink.write(data, "TestLogger")

            # Verify add_scalar calls
            expected_calls = [
                (("TestLogger/loss", 0.3), {"global_step": 5}),
                (("TestLogger/accuracy", 0.8), {"global_step": 5}),
            ]

            assert mock_writer.add_scalar.call_count == 2
            for call, (expected_args, expected_kwargs) in zip(
                mock_writer.add_scalar.call_args_list, expected_calls
            ):
                assert call.args == expected_args
                assert call.kwargs == expected_kwargs


class TestMultiSinkLogger:
    """Test logger with multiple sinks."""

    def test_logger_with_multiple_sinks(self):
        """Test logger functioning with multiple sinks."""
        # Create mock sinks
        sink1 = Mock()
        sink2 = Mock()
        sink3 = Mock()

        logger = TestLogger(name="MultiSinkTest", sinks=[sink1, sink2, sink3])

        # Log some data
        logger.step.log(1)
        logger.value.log(0.5)
        logger.value.log(0.7)

        # Flush should call all sinks
        entry = logger.flush()

        # Verify all sinks received the data
        for sink in [sink1, sink2, sink3]:
            sink.write.assert_called_once()
            args, kwargs = sink.write.call_args
            data, logger_name = args

            assert logger_name == "MultiSinkTest"
            assert data["step"] == 1
            assert data["value"] == 0.6  # Mean of 0.5 and 0.7
            assert "_timestamp" in data

    def test_sink_error_handling(self):
        """Test that sink errors don't crash logging."""
        # Create a sink that will raise an exception
        bad_sink = Mock()
        bad_sink.write.side_effect = Exception("Sink error")

        good_sink = Mock()

        logger = TestLogger(sinks=[bad_sink, good_sink])
        logger.step.log(1)
        logger.value.log(0.5)

        # Should complete without raising exception
        with pytest.warns(RuntimeWarning, match="Sink Mock failed to write"):
            entry = logger.flush()

        # Good sink should still be called
        good_sink.write.assert_called_once()

        # Bad sink should have been attempted
        bad_sink.write.assert_called_once()

    def test_add_remove_sinks(self):
        """Test adding and removing sinks dynamically."""
        logger = TestLogger()

        sink1 = Mock()
        sink2 = Mock()

        # Add sinks
        logger.add_sink(sink1)
        logger.add_sink(sink2)
        assert len(logger.sinks) == 2

        # Test logging goes to both
        logger.step.log(1)
        logger.value.log(0.5)
        logger.flush()

        sink1.write.assert_called_once()
        sink2.write.assert_called_once()

        # Remove one sink
        assert logger.remove_sink(sink1) is True
        assert len(logger.sinks) == 1

        # Remove non-existent sink
        assert logger.remove_sink(Mock()) is False

        # Reset mocks and test again
        sink1.reset_mock()
        sink2.reset_mock()

        logger.step.log(2)
        logger.value.log(0.8)
        logger.flush()

        # Only sink2 should be called
        sink1.write.assert_not_called()
        sink2.write.assert_called_once()

    def test_context_manager(self):
        """Test logger as context manager."""
        sink = Mock()

        with TestLogger(sinks=[sink]) as logger:
            logger.step.log(1)
            logger.value.log(0.5)
            logger.flush()

        # Sink should have been closed
        sink.close.assert_called_once()

    def test_close_method(self):
        """Test explicit close method."""
        sink1 = Mock()
        sink2 = Mock()

        logger = TestLogger(sinks=[sink1, sink2])
        logger.close()

        # Both sinks should be closed
        sink1.close.assert_called_once()
        sink2.close.assert_called_once()

        # Sinks list should be cleared
        assert len(logger.sinks) == 0

    def test_close_with_sink_error(self):
        """Test close method with sink errors."""
        bad_sink = Mock()
        bad_sink.close.side_effect = Exception("Close error")

        good_sink = Mock()

        logger = TestLogger(sinks=[bad_sink, good_sink])

        with pytest.warns(RuntimeWarning, match="Error closing sink"):
            logger.close()

        # Both should have been attempted
        bad_sink.close.assert_called_once()
        good_sink.close.assert_called_once()

        # Sinks should still be cleared
        assert len(logger.sinks) == 0


class TestSinkIntegration:
    """Integration tests with real sinks."""

    def test_console_and_file_integration(self):
        """Test real console and file sink integration."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
            temp_path = Path(f.name)

        try:
            console_sink = ConsoleSink(show_timestamp=False, show_logger_name=True)
            file_sink = FileSink(temp_path, mode="w")

            logger = TestLogger(name="IntegrationTest", sinks=[console_sink, file_sink])

            # Log multiple entries
            for i in range(3):
                logger.step.log(i)
                logger.value.log(0.5 + i * 0.1)
                logger.flush()

            logger.close()

            # Verify file contents
            with open(temp_path, "r") as f:
                lines = f.readlines()

            assert len(lines) == 3

            for i, line in enumerate(lines):
                data = json.loads(line.strip())
                assert data["_logger"] == "IntegrationTest"
                assert data["step"] == i
                assert data["value"] == 0.5 + i * 0.1

        finally:
            temp_path.unlink()
