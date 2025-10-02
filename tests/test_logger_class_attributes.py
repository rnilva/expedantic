"""Tests for logger class attribute configuration functionality."""

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


class TestLoggerClassAttributes:
    """Test the class attribute configuration functionality."""

    def test_basic_class_sinks_attribute(self):
        """Test basic _sinks class attribute."""
        console_sink = ConsoleSink()

        class TestLogger(LoggerBase):
            _sinks = [console_sink]
            value: Field[int]

        logger = TestLogger()
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is console_sink

    def test_basic_class_name_attribute(self):
        """Test basic _name class attribute."""

        class TestLogger(LoggerBase):
            _name = "CustomName"
            value: Field[int]

        logger = TestLogger()
        assert logger.name == "CustomName"

    def test_both_class_attributes(self):
        """Test using both _sinks and _name class attributes."""
        sink = ConsoleSink()

        class TestLogger(LoggerBase):
            _sinks = [sink]
            _name = "BothAttributes"
            value: Field[int]

        logger = TestLogger()
        assert logger.name == "BothAttributes"
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is sink

    def test_multiple_class_sinks(self):
        """Test class attribute with multiple sinks."""
        console_sink = ConsoleSink()
        file_sink = FileSink("test.log")

        class TestLogger(LoggerBase):
            _sinks = [console_sink, file_sink]
            value: Field[int]

        logger = TestLogger()
        assert len(logger.sinks) == 2
        assert logger.sinks[0] is console_sink
        assert logger.sinks[1] is file_sink

    def test_class_attributes_create_copy(self):
        """Test that class attributes create a copy to avoid shared state."""
        sink = Mock()

        class TestLogger(LoggerBase):
            _sinks = [sink]
            value: Field[int]

        logger1 = TestLogger()
        logger2 = TestLogger()

        # Both should have the same sink but different lists
        assert logger1.sinks[0] is sink
        assert logger2.sinks[0] is sink
        assert logger1.sinks is not logger2.sinks  # Different list objects

    def test_explicit_overrides_class_attributes(self):
        """Test that explicit constructor parameters override class attributes."""
        class_sink = ConsoleSink()
        explicit_sink = FileSink("explicit.log")

        class TestLogger(LoggerBase):
            _sinks = [class_sink]
            _name = "ClassName"
            value: Field[int]

        # Default should use class attributes
        default_logger = TestLogger()
        assert default_logger.name == "ClassName"
        assert len(default_logger.sinks) == 1
        assert default_logger.sinks[0] is class_sink

        # Explicit should override class attributes
        explicit_logger = TestLogger(name="ExplicitName", sinks=[explicit_sink])
        assert explicit_logger.name == "ExplicitName"
        assert len(explicit_logger.sinks) == 1
        assert explicit_logger.sinks[0] is explicit_sink

    def test_class_attributes_override_decorators(self):
        """Test that class attributes take precedence over decorators."""
        decorator_sink = ConsoleSink()
        class_sink = FileSink("class.log")

        @logger_sinks([decorator_sink])
        @logger_name("DecoratorName")
        class TestLogger(LoggerBase):
            _sinks = [class_sink]
            _name = "ClassName"
            value: Field[int]

        logger = TestLogger()
        # Class attributes should override decorators
        assert logger.name == "ClassName"
        assert len(logger.sinks) == 1
        assert logger.sinks[0] is class_sink

    def test_partial_class_attributes_with_decorators(self):
        """Test mixing class attributes with decorators."""
        decorator_sink = ConsoleSink()
        class_sink = FileSink("class.log")

        # Only decorator name, class sinks
        @logger_name("DecoratorName")
        class TestLogger1(LoggerBase):
            _sinks = [class_sink]
            value: Field[int]

        logger1 = TestLogger1()
        assert logger1.name == "DecoratorName"  # From decorator
        assert logger1.sinks[0] is class_sink  # From class attribute

        # Only decorator sinks, class name
        @logger_sinks([decorator_sink])
        class TestLogger2(LoggerBase):
            _name = "ClassName"
            value: Field[int]

        logger2 = TestLogger2()
        assert logger2.name == "ClassName"  # From class attribute
        assert logger2.sinks[0] is decorator_sink  # From decorator

    def test_no_class_attributes_uses_defaults(self):
        """Test fallback behavior when no class attributes are set."""

        class PlainLogger(LoggerBase):
            value: Field[int]

        logger = PlainLogger()
        assert logger.name == "PlainLogger"  # Class name
        assert len(logger.sinks) == 1
        assert isinstance(logger.sinks[0], ConsoleSink)  # Default ConsoleSink

    def test_none_class_attributes_ignored(self):
        """Test that None class attributes are ignored."""

        class TestLogger(LoggerBase):
            _sinks = None
            _name = None
            value: Field[int]

        logger = TestLogger()
        assert logger.name == "TestLogger"  # Falls back to class name
        assert len(logger.sinks) == 1
        assert isinstance(logger.sinks[0], ConsoleSink)  # Default ConsoleSink

    def test_empty_class_sinks_list(self):
        """Test class attribute with empty sinks list."""

        class TestLogger(LoggerBase):
            _sinks = []
            value: Field[int]

        logger = TestLogger()
        assert len(logger.sinks) == 0

    def test_class_attribute_inheritance(self):
        """Test class attribute behavior with inheritance."""
        base_sink = ConsoleSink()
        child_sink = FileSink("child.log")

        class BaseLogger(LoggerBase):
            _sinks = [base_sink]
            _name = "BaseName"
            value: Field[int]

        class ChildLogger(BaseLogger):
            _sinks = [child_sink]
            _name = "ChildName"
            extra: Field[str]

        # Base should use base attributes
        base_logger = BaseLogger()
        assert base_logger.name == "BaseName"
        assert base_logger.sinks[0] is base_sink

        # Child should use child attributes (overrides base)
        child_logger = ChildLogger()
        assert child_logger.name == "ChildName"
        assert child_logger.sinks[0] is child_sink

    def test_precedence_order(self):
        """Test the complete precedence order: explicit > class > decorator > default."""
        decorator_sink = Mock()
        class_sink = Mock()
        explicit_sink = Mock()

        @logger_sinks([decorator_sink])
        @logger_name("DecoratorName")
        class TestLogger(LoggerBase):
            _sinks = [class_sink]
            _name = "ClassName"
            value: Field[int]

        # Test all combinations of precedence
        logger1 = TestLogger()  # Should use class attributes
        assert logger1.name == "ClassName"
        assert logger1.sinks[0] is class_sink

        logger2 = TestLogger(name="ExplicitName")  # Explicit name, class sinks
        assert logger2.name == "ExplicitName"
        assert logger2.sinks[0] is class_sink

        logger3 = TestLogger(sinks=[explicit_sink])  # Class name, explicit sinks
        assert logger3.name == "ClassName"
        assert logger3.sinks[0] is explicit_sink

        logger4 = TestLogger(name="ExplicitName", sinks=[explicit_sink])  # All explicit
        assert logger4.name == "ExplicitName"
        assert logger4.sinks[0] is explicit_sink


class TestClassAttributeIntegration:
    """Integration tests with real functionality."""

    def test_class_attribute_logger_functionality(self):
        """Test that class attribute loggers work correctly with logging."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".jsonl") as f:
            temp_path = Path(f.name)

        try:
            console_sink = ConsoleSink(show_timestamp=False, show_logger_name=False)
            file_sink = FileSink(temp_path, mode="w")

            class TestLogger(LoggerBase):
                _sinks = [console_sink, file_sink]
                _name = "IntegrationTest"

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

    def test_class_attributes_with_context_manager(self):
        """Test class attribute logger works with context manager."""
        sink = Mock()

        class TestLogger(LoggerBase):
            _sinks = [sink]
            value: Field[int]

        with TestLogger() as logger:
            logger.value.log(42)
            logger.flush()

        # Sink should have been closed
        sink.close.assert_called_once()

    def test_mixed_configuration_approaches(self):
        """Test that different configuration approaches can coexist."""

        # Decorator-based logger
        @logger_sinks([ConsoleSink()])
        class DecoratorLogger(LoggerBase):
            value: Field[int]

        # Class attribute logger
        class ClassAttributeLogger(LoggerBase):
            _sinks = [ConsoleSink()]
            value: Field[int]

        # Plain logger
        class PlainLogger(LoggerBase):
            value: Field[int]

        # All should work independently
        dec_logger = DecoratorLogger()
        class_logger = ClassAttributeLogger()
        plain_logger = PlainLogger()

        assert len(dec_logger.sinks) == 1
        assert len(class_logger.sinks) == 1
        assert len(plain_logger.sinks) == 1

        # Test logging works for all
        for logger in [dec_logger, class_logger, plain_logger]:
            logger.value.log(100)
            entry = logger.flush()
            assert entry["value"] == 100
