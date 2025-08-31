"""Core logger functionality.

This module contains the LoggerBase class and related functionality
for creating type-safe, structured loggers.
"""

import inspect
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import get_type_hints, get_origin

import polars as pl

from .fields import FieldBase, SupportedTypes
from .sinks import SinkProtocol


class LoggerBase:
    """Base class for creating type-safe, structured loggers.

    LoggerBase uses Python type annotations to automatically create field instances
    and manage data collection. Subclasses define their schema by simply declaring
    class attributes with field type annotations.

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

    def __init__(
        self, name: str | None = None, sinks: list[SinkProtocol] | None = None
    ) -> None:
        """Initialize the logger with field schema and optional sinks.

        Args:
            name: Optional name for this logger instance
            sinks: List of sinks to receive flushed data. If None, data is only stored internally.
        """
        self.name = name or self.__class__.__name__
        self.sinks: list[SinkProtocol] = sinks or []

        # Extract field schema from type annotations
        self.schema = {
            k: v
            for k, v in get_type_hints(self).items()
            if (inspect.isclass(v) and issubclass(v, FieldBase))
            or ((o := get_origin(v)) is not None and issubclass(o, FieldBase))
        }
        # Instantiate field objects
        for k, v in self.schema.items():
            setattr(self, k, v())

        self.data: list[dict[str, SupportedTypes]] = []

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

    def to_dataframe(self) -> pl.DataFrame:
        """Convert all logged entries to a Polars DataFrame.

        Returns:
            pl.DataFrame: A DataFrame with one row per flush() call,
                         columns for each field, and a '_timestamp' column
        """
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
