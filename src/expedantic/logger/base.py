"""Core logger functionality.

This module contains the LoggerBase class and related functionality
for creating type-safe, structured loggers.
"""

import inspect
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, get_origin

from .fields import FieldBase, SupportedTypes
from .sinks import SinkProtocol

if TYPE_CHECKING:
    import polars as pl


def _require_polars():
    """Import polars on demand, with an actionable error if it is missing.

    polars is an optional dependency: it is only needed by the DataFrame and
    Arrow IPC helpers, not by logging itself.
    """
    try:
        import polars as pl
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "polars is required for DataFrame export but is not installed. "
            "Install it with: pip install 'expedantic[dataframe]'"
        ) from e
    return pl


class LoggerBase:
    """Base class for creating type-safe, structured loggers.

    LoggerBase uses Python type annotations to automatically create field instances
    and manage data collection. Subclasses define their schema by simply declaring
    class attributes with field type annotations.

    Configuration can be done in three ways:

    1. Using decorators (recommended for complex setups):
        @logger_sinks([ConsoleSink(), FileSink("log.jsonl")])
        @logger_name("MyLogger")
        class MyLogger(LoggerBase):
            iteration: Field[int]
            loss: MeanField

    2. Using class attributes (great for IDE autocompletion):
        class MyLogger(LoggerBase):
            _sinks: list[SinkProtocol] = [ConsoleSink(), FileSink("log.jsonl")]
            _name: str = "MyLogger"

            iteration: Field[int]
            loss: MeanField

    3. Using constructor parameters (runtime configuration):
        logger = MyLogger(name="Custom", sinks=[ConsoleSink()])

    Example:
        class MyLogger(LoggerBase):
            iteration: Field[int]
            loss: MeanField
            accuracy: MaxField[float]

        logger = MyLogger()
        logger.iteration.log(1)
        logger.loss.log(0.5)
        entry = logger.flush()  # Computes aggregations and stores entry
    """

    # Configuration class attributes (for IDE autocompletion)
    # Users can override these in their subclasses
    _sinks: list[SinkProtocol] | None = None
    _name: str | None = None

    def __init__(
        self, name: str | None = None, sinks: list[SinkProtocol] | None = None
    ) -> None:
        """Initialize the logger with field schema and optional sinks.

        Args:
            name: Optional name for this logger instance. If None, uses class attribute
                  _name, decorator-defined name, or falls back to class name.
            sinks: List of sinks to receive flushed data. If None, uses class attribute
                   _sinks, decorator-defined sinks, or defaults to ConsoleSink.
        """
        # Determine name to use (precedence: explicit > class attribute > decorator > class name)
        if name is not None:
            self.name = name
        else:
            # Check for class-level _name attribute
            class_name = getattr(self.__class__, "_name", None)
            if class_name is not None:
                self.name = class_name
            else:
                # Check for decorator-defined default name
                default_name = getattr(self.__class__, "_default_name", None)
                self.name = default_name or self.__class__.__name__

        # Determine sinks to use (precedence: explicit > class attribute > decorator > default)
        if sinks is not None:
            # Explicit sinks provided - use them
            self.sinks: list[SinkProtocol] = sinks
        else:
            # Check for class-level _sinks attribute
            class_sinks = getattr(self.__class__, "_sinks", None)
            if class_sinks is not None:
                # Use class-defined sinks (create a copy to avoid shared state)
                self.sinks = list(class_sinks)
            else:
                # Check for decorator-defined default sinks
                default_sinks = getattr(self.__class__, "_default_sinks", None)
                if default_sinks is not None:
                    # Use decorator-defined sinks (create a copy to avoid shared state)
                    self.sinks = list(default_sinks)
                else:
                    # Default to console sink for convenience
                    from .sinks import ConsoleSink

                    self.sinks = [ConsoleSink()]

        # Extract field schema from type annotations, excluding configuration attributes.
        self.schema = self._resolve_schema()
        # Instantiate field objects
        for k, v in self.schema.items():
            setattr(self, k, v())

        self.data: list[dict[str, SupportedTypes]] = []

    def _resolve_schema(self) -> dict:
        """Resolve the field schema from this class's (and its bases') annotations.

        Annotations are resolved separately for each class. ``inspect.get_annotations``
        ignores inherited annotations and evaluates each class against its own module
        globals, so both stringized annotations and Python 3.14's deferred annotations
        are handled without trying to resolve a base class in the subclass's namespace.

        Bases are walked in reverse MRO order so the most-derived class's annotation
        wins on name collisions, matching normal attribute-override semantics.
        """
        schema: dict = {}
        for cls in reversed(type(self).__mro__):
            if cls is object:
                continue
            try:
                hints = inspect.get_annotations(cls, eval_str=True)
            except Exception:
                # Skip a class whose hints cannot be resolved rather than failing
                # the whole logger. Other classes in the hierarchy remain usable.
                continue
            for k, v in hints.items():
                if k in ("_sinks", "_name"):  # Skip configuration attributes
                    continue
                origin = get_origin(v)
                if (inspect.isclass(v) and issubclass(v, FieldBase)) or (
                    inspect.isclass(origin) and issubclass(origin, FieldBase)
                ):
                    schema[k] = v
        return schema

    def flush(self):
        """Compute field aggregations and store the results as a data entry.

        This method collects the current value from each field, computes any
        necessary aggregations, and stores the results. Fields are reset after
        flushing (for reducible fields, this means clearing their value lists).

        Returns:
            dict: A dictionary mapping field names to their computed values.
                 Also includes '_timestamp' with the flush time.
        """
        items: dict[str, SupportedTypes] = {}
        for k in self.schema:
            field: FieldBase = getattr(self, k)
            value = field.value
            # Include all values except None (empty collections are valid)
            if value is not None:
                items[k] = field.value

        if items:
            # Add automatic timestamp
            items["_timestamp"] = datetime.now()
            # Stamp the logger name into the canonical row so EVERY sink and
            # to_dataframe()/save() see identical keys. Previously FileSink injected
            # a "_logger" key on its own, so the JSONL carried "_logger" while the
            # in-memory data / parquet did not — mismatched schemas across outputs.
            if self.name is not None:
                items["_logger"] = self.name
            self.data.append(items)

            # Send to all sinks
            for sink in self.sinks:
                try:
                    sink.write(items, self.name)
                except Exception as e:
                    # Don't let sink errors crash logging
                    import warnings

                    warnings.warn(
                        f"Sink {sink.__class__.__name__} failed to write: {e}",
                        RuntimeWarning,
                    )

        # Reset fields after flushing
        for k in self.schema:
            field: FieldBase = getattr(self, k)
            field.reset()

        return items

    def to_dataframe(self) -> "pl.DataFrame":
        """Convert all logged entries to a Polars DataFrame.

        Requires the optional ``dataframe`` extra (``pip install
        'expedantic[dataframe]'``).

        Returns:
            pl.DataFrame: A DataFrame with one row per flush() call,
                         columns for each field, and a '_timestamp' column
        """
        pl = _require_polars()
        return pl.DataFrame(self.data)

    def save(self, path: str | Path | BytesIO):
        """Save logged data to a file in Apache Arrow IPC format.

        Args:
            path: File path or BytesIO buffer to write to
        """
        df = self.to_dataframe()
        df.write_ipc(path)

    def __len__(self):
        """Get the number of entries (flush calls) in this logger.

        Returns:
            int: The number of data entries
        """
        return len(self.data)

    def add_field(self, name: str, field_type) -> FieldBase:
        """Register a new field on this logger instance at runtime.

        Use this for fields whose names/count are not known at class-definition
        time (e.g. one column per layer of a net). It mirrors the schema setup done
        in ``__init__``: it records ``field_type`` in ``self.schema``, instantiates the
        field, and binds it as ``self.<name>`` so ``self.<name>.log(...)`` works and the
        column appears in ``flush()`` rows / ``to_dataframe()``.

        Args:
            name: The field (column) name to register.
            field_type: A ``FieldBase`` subclass or parameterized field type
                (e.g. ``Field``, ``Field[int]``, ``MeanField``).

        Returns:
            FieldBase: The instantiated field, the same object as ``getattr(self, name)``.

        Raises:
            TypeError: If ``field_type`` is not a field type.
            ValueError: If ``name`` is already registered with a *different* type.

        Note:
            Idempotent: re-adding the same ``name`` with the same ``field_type`` returns
            the existing field and does not reset its accumulated values.
        """
        # Validate that field_type really is a field (class or parameterized generic).
        origin = get_origin(field_type)
        is_field_type = (
            inspect.isclass(field_type) and issubclass(field_type, FieldBase)
        ) or (origin is not None and issubclass(origin, FieldBase))
        if not is_field_type:
            raise TypeError(
                f"add_field expected a FieldBase subclass or parameterized field type, "
                f"got {field_type!r}"
            )

        if name in self.schema:
            if self.schema[name] is field_type:
                # Idempotent no-op: keep the existing field (and its state).
                return getattr(self, name)
            raise ValueError(
                f"Field {name!r} already registered as {self.schema[name]!r}; "
                f"cannot re-register as {field_type!r}"
            )

        self.schema[name] = field_type
        field = field_type()
        setattr(self, name, field)
        return field

    def add_sink(self, sink: SinkProtocol) -> None:
        """Add a sink to receive flushed data.

        Args:
            sink: The sink instance to add
        """
        self.sinks.append(sink)

    def remove_sink(self, sink: SinkProtocol) -> bool:
        """Remove a sink from the logger.

        Args:
            sink: The sink instance to remove

        Returns:
            bool: True if sink was found and removed, False otherwise
        """
        try:
            self.sinks.remove(sink)
            return True
        except ValueError:
            return False

    def close(self) -> None:
        """Close all sinks and release resources.

        This should be called when done with the logger to ensure
        proper cleanup of file handles, network connections, etc.
        """
        for sink in self.sinks:
            try:
                sink.close()
            except Exception as e:
                import warnings

                warnings.warn(
                    f"Error closing sink {sink.__class__.__name__}: {e}", RuntimeWarning
                )
        self.sinks.clear()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes all sinks."""
        self.close()
