"""Field types for the logger system.

This module contains all the field implementations that handle different
aggregation patterns for logged data.
"""

import statistics
import threading
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import (
    Any,
    Generic,
    Protocol,
    TypeAlias,
    TypeVar,
    TYPE_CHECKING,
)

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
            raise ValueError(
                "Cannot log None value to Field. Use Optional types if None is expected."
            )
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
            return statistics.fmean(self.values)
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
            # pstdev, not stdev: numpy's std defaults to ddof=0 (population).
            return float(statistics.pstdev(self.values))
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Cannot compute standard deviation of non-numeric values: {e}"
            )


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
            return float(statistics.median(self.values))
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
