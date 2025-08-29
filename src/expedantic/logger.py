import inspect
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
    get_type_hints,
    get_origin,
)

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparisonT, SupportsAdd
else:
    _T_co = TypeVar("_T_co", covariant=True)
    _T_contra = TypeVar("_T_contra", contravariant=True)

    class SupportsDunderLT(Protocol[_T_contra]):
        def __lt__(self, other: _T_contra, /) -> bool: ...

    class SupportsDunderGT(Protocol[_T_contra]):
        def __gt__(self, other: _T_contra, /) -> bool: ...

    class SupportsDunderLE(Protocol[_T_contra]):
        def __le__(self, other: _T_contra, /) -> bool: ...

    class SupportsDunderGE(Protocol[_T_contra]):
        def __ge__(self, other: _T_contra, /) -> bool: ...

    class SupportsAllComparisons(
        SupportsDunderLT[Any],
        SupportsDunderGT[Any],
        SupportsDunderLE[Any],
        SupportsDunderGE[Any],
        Protocol,
    ): ...

    SupportsRichComparison: TypeAlias = SupportsDunderLT[Any] | SupportsDunderGT[Any]
    SupportsRichComparisonT = TypeVar(
        "SupportsRichComparisonT", bound=SupportsRichComparison
    )

    class SupportsAdd(Protocol[_T_contra, _T_co]):
        def __add__(self, x: _T_contra, /) -> _T_co: ...


SupportedTypes = (
    int
    | float
    | bool
    | str
    | date
    | time
    | datetime
    | timedelta
    | list[Any]
    | tuple[Any, ...]
    | bytes
    | object
    | Decimal
    | None
)

T = TypeVar("T", bound=SupportedTypes)
TSupportsAdd = TypeVar(
    "TSupportsAdd", bound=int | float | bool | date | timedelta | list[Any] | tuple[Any]
)


class FieldBase(ABC, Generic[T]):
    """Abstract base class for all field types.
    
    Fields are responsible for storing and aggregating logged values.
    Each field type implements different aggregation strategies.
    """
    
    def log(self, value: T): 
        """Log a value to this field.
        
        Args:
            value: The value to log, must match the field's type T
        """
        ...

    @property
    @abstractmethod
    def value(self) -> Any: 
        """Get the current aggregated value of this field.
        
        Returns:
            The aggregated result based on the field's strategy
        """
        ...
        
    def reset(self):
        """Reset the field to its initial state after flush.
        
        This method should be overridden by subclasses that need
        to clear accumulated state.
        """
        pass


class ReducibleFieldBase(FieldBase[T]):
    """Base class for fields that collect multiple values for aggregation.
    
    This class stores all logged values in a list and provides them
    to subclasses for various reduction operations (mean, max, etc.).
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.values: list[T] = []

    def log(self, value: T):
        """Store a value for later aggregation.
        
        Args:
            value: The value to store
            
        Raises:
            ValueError: If value is None
        """
        if value is None:
            raise ValueError(f"Cannot log None value to {self.__class__.__name__}")
        self.values.append(value)
        
    def reset(self):
        """Clear all accumulated values."""
        self.values.clear()


class Field(FieldBase[T]):
    """A field that stores only the most recently logged value.
    
    This is useful for tracking current state values like iteration number,
    learning rate, or any metric where only the latest value matters.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._value: T | None = None

    def log(self, value: T):
        """Store the value, overwriting any previous value.
        
        Args:
            value: The value to store
            
        Raises:
            ValueError: If value is None (use explicit None handling if needed)
        """
        if value is None:
            raise ValueError("Cannot log None value to Field. Use Optional types if None is expected.")
        self._value = value

    @property
    def value(self) -> T | None:
        """Get the most recently logged value.
        
        Returns:
            The last logged value, or None if nothing has been logged
        """
        return self._value


class MeanField(ReducibleFieldBase[float]):
    """A field that computes the arithmetic mean of all logged values.
    
    Useful for tracking average metrics like loss, accuracy, or timing measurements
    across multiple batches or iterations.
    """
    
    @property
    def value(self) -> float | None:
        """Get the mean of all logged values.
        
        Returns:
            The arithmetic mean, or None if no values have been logged
            
        Raises:
            ValueError: If any logged values are not numeric
        """
        if not self.values:
            return None
        try:
            return np.mean(self.values).item()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot compute mean of non-numeric values: {e}")


class MaxField(ReducibleFieldBase[SupportsRichComparisonT]):
    """A field that tracks the maximum value among all logged values.
    
    Useful for tracking peak performance metrics like best accuracy
    or highest validation score.
    """
    
    @property
    def value(self):
        """Get the maximum of all logged values.
        
        Returns:
            The maximum value, or None if no values have been logged
            
        Raises:
            TypeError: If values cannot be compared
        """
        if not self.values:
            return None
        try:
            return max(self.values)
        except TypeError as e:
            raise TypeError(f"Cannot compare values to find maximum: {e}")


class MinField(ReducibleFieldBase[SupportsRichComparisonT]):
    """A field that tracks the minimum value among all logged values.
    
    Useful for tracking best performance metrics like lowest loss
    or minimum error rate.
    """
    
    @property
    def value(self):
        """Get the minimum of all logged values.
        
        Returns:
            The minimum value, or None if no values have been logged
            
        Raises:
            TypeError: If values cannot be compared
        """
        if not self.values:
            return None
        try:
            return min(self.values)
        except TypeError as e:
            raise TypeError(f"Cannot compare values to find minimum: {e}")


class StdField(ReducibleFieldBase[float]):
    """A field that computes the standard deviation of all logged values.
    
    Useful for tracking variability in metrics like loss stability
    or measurement precision.
    """
    
    @property
    def value(self) -> float | None:
        """Get the standard deviation of all logged values.
        
        Returns:
            The standard deviation, or None if no values have been logged
            
        Raises:
            ValueError: If any logged values are not numeric
        """
        if not self.values:
            return None
        try:
            return np.std(self.values).item()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot compute standard deviation of non-numeric values: {e}")


class MedianField(ReducibleFieldBase[float]):
    """A field that computes the median of all logged values.
    
    Useful for robust central tendency measurement that's less sensitive
    to outliers than mean.
    """
    
    @property
    def value(self) -> float | None:
        """Get the median of all logged values.
        
        Returns:
            The median value, or None if no values have been logged
            
        Raises:
            ValueError: If any logged values are not numeric
        """
        if not self.values:
            return None
        try:
            return np.median(self.values).item()
        except (TypeError, ValueError) as e:
            raise ValueError(f"Cannot compute median of non-numeric values: {e}")


class SumField(ReducibleFieldBase[TSupportsAdd]):
    """A field that computes the sum of all logged values.
    
    Useful for accumulating values like total training time,
    total number of samples processed, or cumulative costs.
    """
    
    @property
    def value(self):
        """Get the sum of all logged values.
        
        Returns:
            The sum of all values, starting from 0
        """
        return sum(self.values, 0)


class ListField(ReducibleFieldBase[T]):
    """A field that collects all logged values in a list.
    
    Useful for storing sequences of values like error messages,
    file paths, or any data that needs to be kept in full detail.
    """
    
    @property
    def value(self) -> list[T] | None:
        """Get all logged values as a list.
        
        Returns:
            A list containing all logged values in order, or None if empty
        """
        return self.values.copy() if self.values else None


class CountField(FieldBase[int]):
    """A field that counts the number of times log() has been called.
    
    The actual value passed to log() is ignored; this field only tracks
    the count of logging events. Useful for counting errors, events, or
    any occurrence-based metrics.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._count: int = 0

    def log(self, value: Any):
        """Increment the counter (value is ignored).
        
        Args:
            value: Ignored, any value can be passed
        """
        self._count += 1

    @property
    def value(self) -> int:
        """Get the current count.
        
        Returns:
            The number of times log() has been called
        """
        return self._count


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
    
    def __init__(self) -> None:
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
            items['_timestamp'] = datetime.now()
            self.data.append(items)
            
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


class TrainingLogger(LoggerBase):
    """Example logger for machine learning training loops.
    
    This logger demonstrates common patterns for tracking training metrics:
    - Current state values (iteration, epoch, learning_rate)
    - Averaged metrics (loss, val_loss)  
    - Peak performance tracking (accuracy)
    - Accumulated values (batch_time)
    - Event collection (messages)
    """

    iteration: Field[int]           # Current training iteration
    epoch: Field[int]               # Current epoch number
    loss: MeanField                 # Average training loss per epoch
    val_loss: MeanField             # Average validation loss per epoch  
    learning_rate: Field[float]     # Current learning rate
    messages: ListField[str]        # Collected log messages
    accuracy: MaxField[float]       # Best accuracy achieved in epoch
    batch_time: SumField[float]     # Total time spent on batches



