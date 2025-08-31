"""Tests for logger decorators functionality."""

import tempfile
from pathlib import Path
from unittest.mock import Mock
import pytest

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    ConsoleSink,
    FileSink,
    logger_sinks,
    logger_name,
)


class TestLoggerSinksDecorator:
    """Test the @logger_sinks decorator functionality."""

    def test_basic_decorator_usage(self):
        """Test basic @logger_sinks decorator."""
        console_sink = ConsoleSink()

        @logger_sinks([console_sink])
        class TestLogger(LoggerBase):
            value: Field[int]

        # Class should have _default_sinks attribute
        assert hasattr(TestLogger, "_default_sinks")
        assert TestLogger._default_sinks == [console_sink]

        # Instance should use decorator-defined sinks
        logger = TestLogger()
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is console_sink

    def test_multiple_sinks_decorator(self):
        """Test decorator with multiple sinks."""
        console_sink = ConsoleSink()
        file_sink = FileSink("test.log")

        @logger_sinks([console_sink, file_sink])
        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger()
        assert len(logger.sinks) == 2
        assert logger.sinks[0] is console_sink
        assert logger.sinks[1] is file_sink

    def test_decorator_creates_copy(self):
        """Test that decorator creates a copy of sinks to avoid shared state."""
        sink = Mock()

        @logger_sinks([sink])
        class TestLogger(LoggerBase):
            value: Field[int]

        logger1 = TestLogger()
        logger2 = TestLogger()

        # Both should have the same sink type but different lists
        assert logger1.sinks[0] is sink
        assert logger2.sinks[0] is sink
        assert logger1.sinks is not logger2.sinks  # Different list objects

    def test_explicit_sinks_override_decorator(self):
        """Test that explicit sinks parameter overrides decorator."""
        decorator_sink = ConsoleSink()
        explicit_sink = FileSink("explicit.log")

        @logger_sinks([decorator_sink])
        class TestLogger(LoggerBase):
            value: Field[int]

        # Default should use decorator sinks
        default_logger = TestLogger()
        assert len(default_logger.sinks) == 1
        assert default_logger.sinks[0] is decorator_sink

        # Explicit should override decorator
        explicit_logger = TestLogger(sinks=[explicit_sink])
        assert len(explicit_logger.sinks) == 1
        assert explicit_logger.sinks[0] is explicit_sink

    def test_decorator_validation_non_list(self):
        """Test that decorator validates input is a list."""
        with pytest.raises(TypeError, match="expects a list"):

            @logger_sinks("not a list")  # type: ignore
            class TestLogger(LoggerBase):
                value: Field[int]

    def test_decorator_validation_sink_protocol(self):
        """Test that decorator validates sinks implement protocol."""
        invalid_sink = "not a sink"

        with pytest.raises(TypeError, match="must implement the SinkProtocol"):

            @logger_sinks([invalid_sink])  # type: ignore
            class TestLogger(LoggerBase):
                value: Field[int]

    def test_no_decorator_uses_default_console(self):
        """Test that loggers without decorators get default ConsoleSink."""

        class PlainLogger(LoggerBase):
            value: Field[int]

        logger = PlainLogger()
        assert len(logger.sinks) == 1
        assert isinstance(logger.sinks[0], ConsoleSink)

    def test_empty_decorator_sinks_list(self):
        """Test decorator with empty sinks list."""

        @logger_sinks([])
        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger()
        assert len(logger.sinks) == 0

    def test_decorator_inheritance(self):
        """Test decorator behavior with class inheritance."""
        base_sink = ConsoleSink()
        child_sink = FileSink("child.log")

        @logger_sinks([base_sink])
        class BaseLogger(LoggerBase):
            value: Field[int]

        @logger_sinks([child_sink])
        class ChildLogger(BaseLogger):
            extra: Field[str]

        # Base should use base sinks
        base_logger = BaseLogger()
        assert len(base_logger.sinks) == 1
        assert base_logger.sinks[0] is base_sink

        # Child should use child sinks (overrides base)
        child_logger = ChildLogger()
        assert len(child_logger.sinks) == 1
        assert child_logger.sinks[0] is child_sink


class TestLoggerNameDecorator:
    """Test the @logger_name decorator functionality."""

    def test_basic_name_decorator(self):
        """Test basic @logger_name decorator."""

        @logger_name("CustomName")
        class TestLogger(LoggerBase):
            value: Field[int]

        # Class should have _default_name attribute
        assert hasattr(TestLogger, "_default_name")
        assert TestLogger._default_name == "CustomName"

        # Instance should use decorator-defined name
        logger = TestLogger()
        assert logger.name == "CustomName"

    def test_explicit_name_overrides_decorator(self):
        """Test that explicit name parameter overrides decorator."""

        @logger_name("DecoratorName")
        class TestLogger(LoggerBase):
            value: Field[int]

        # Default should use decorator name
        default_logger = TestLogger()
        assert default_logger.name == "DecoratorName"

        # Explicit should override decorator
        explicit_logger = TestLogger(name="ExplicitName")
        assert explicit_logger.name == "ExplicitName"

    def test_no_decorator_uses_class_name(self):
        """Test that loggers without name decorator use class name."""

        class MyCustomLogger(LoggerBase):
            value: Field[int]

        logger = MyCustomLogger()
        assert logger.name == "MyCustomLogger"

    def test_name_decorator_validation(self):
        """Test that name decorator validates input."""
        with pytest.raises(TypeError, match="expects a string"):

            @logger_name(123)  # type: ignore
            class TestLogger(LoggerBase):
                value: Field[int]

        with pytest.raises(ValueError, match="non-empty string"):

            @logger_name("   ")
            class TestLogger(LoggerBase):
                value: Field[int]

    def test_name_decorator_strips_whitespace(self):
        """Test that name decorator strips whitespace."""

        @logger_name("  SpacedName  ")
        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger()
        assert logger.name == "SpacedName"


class TestCombinedDecorators:
    """Test using both decorators together."""

    def test_both_decorators_combined(self):
        """Test using both @logger_sinks and @logger_name together."""
        sink = ConsoleSink()

        @logger_sinks([sink])
        @logger_name("CombinedLogger")
        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger()
        assert logger.name == "CombinedLogger"
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is sink

    def test_decorator_order_independence(self):
        """Test that decorator order doesn't matter."""
        sink = FileSink("test.log")

        # Order 1: sinks first, then name
        @logger_sinks([sink])
        @logger_name("Order1")
        class Logger1(LoggerBase):
            value: Field[int]

        # Order 2: name first, then sinks
        @logger_name("Order2")
        @logger_sinks([sink])
        class Logger2(LoggerBase):
            value: Field[int]

        logger1 = Logger1()
        logger2 = Logger2()

        assert logger1.name == "Order1"
        assert logger2.name == "Order2"
        assert logger1.sinks[0] is sink
        assert logger2.sinks[0] is sink

    def test_explicit_overrides_both_decorators(self):
        """Test that explicit parameters override both decorators."""
        decorator_sink = ConsoleSink()
        explicit_sink = FileSink("explicit.log")

        @logger_sinks([decorator_sink])
        @logger_name("DecoratorName")
        class TestLogger(LoggerBase):
            value: Field[int]

        logger = TestLogger(name="ExplicitName", sinks=[explicit_sink])
        assert logger.name == "ExplicitName"
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is explicit_sink


class TestDecoratorIntegration:
    """Integration tests with real functionality."""

    def test_decorated_logger_functionality(self):
        """Test that decorated loggers work correctly with logging."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
            temp_path = Path(f.name)

        try:
            console_sink = ConsoleSink(show_timestamp=False, show_logger_name=False)
            file_sink = FileSink(temp_path, mode="w")

            @logger_sinks([console_sink, file_sink])
            @logger_name("IntegrationTest")
            class TestLogger(LoggerBase):
                step: Field[int]
                value: MeanField

            logger = TestLogger()

            # Log some data
            logger.step.log(1)
            logger.value.log(0.5)
            logger.value.log(0.7)
            entry = logger.flush()

            # Verify entry structure
            assert entry["step"] == 1
            assert entry["value"] == 0.6  # Mean of 0.5 and 0.7
            assert "_timestamp" in entry

            # Close to flush file
            logger.close()

            # Verify file contents
            with open(temp_path) as f:
                line = f.readline().strip()
                import json

                data = json.loads(line)
                assert data["_logger"] == "IntegrationTest"
                assert data["step"] == 1
                assert data["value"] == 0.6

        finally:
            temp_path.unlink()

    def test_decorator_with_context_manager(self):
        """Test decorator works with logger context manager."""
        sink = Mock()

        @logger_sinks([sink])
        class TestLogger(LoggerBase):
            value: Field[int]

        with TestLogger() as logger:
            logger.value.log(42)
            logger.flush()

        # Sink should have been closed
        sink.close.assert_called_once()

    def test_multiple_logger_instances_independence(self):
        """Test that multiple instances of decorated loggers are independent."""
        sink1 = Mock()
        sink2 = Mock()

        @logger_sinks([sink1])
        class Logger1(LoggerBase):
            value: Field[int]

        @logger_sinks([sink2])
        class Logger2(LoggerBase):
            value: Field[str]

        logger1 = Logger1()
        logger2 = Logger2()

        logger1.value.log(1)
        logger2.value.log("test")

        logger1.flush()
        logger2.flush()

        # Each sink should only receive its logger's data
        sink1.write.assert_called_once()
        sink2.write.assert_called_once()

        # Check the data passed to each sink
        sink1_data = sink1.write.call_args[0][0]
        sink2_data = sink2.write.call_args[0][0]

        assert sink1_data["value"] == 1
        assert sink2_data["value"] == "test"
