"""Decorators for logger configuration.

This module provides decorators to configure logger behavior in a clean,
declarative way directly on the logger class definition.
"""

from typing import Any, Type, TypeVar
from .sinks import SinkProtocol

# Type variable for the decorated class
T = TypeVar("T")


def logger_sinks(sinks: list[SinkProtocol]) -> callable:
    """Decorator to configure default sinks for a logger class.

    This decorator allows you to specify sinks directly on the logger class
    definition, providing a clean and declarative way to configure logging
    destinations.

    Args:
        sinks: List of sink instances to use as defaults for this logger class

    Returns:
        A decorator function that configures the class with the specified sinks

    Example:
        @logger_sinks([ConsoleSink(), FileSink("training.log")])
        class TrainingLogger(LoggerBase):
            epoch: Field[int]
            loss: MeanField

        # Logger will automatically use the decorated sinks
        logger = TrainingLogger()  # Uses ConsoleSink + FileSink

        # Can still override with explicit sinks
        logger = TrainingLogger(sinks=[FileSink("custom.log")])
    """

    def decorator(cls: Type[T]) -> Type[T]:
        """The actual decorator function.

        Args:
            cls: The logger class to decorate

        Returns:
            The same class with _default_sinks attribute added
        """
        if not isinstance(sinks, list):
            raise TypeError("logger_sinks() expects a list of sink instances")

        if not all(hasattr(sink, "write") and hasattr(sink, "close") for sink in sinks):
            raise TypeError(
                "All sinks must implement the SinkProtocol (write and close methods)"
            )

        # Attach default sinks to the class
        cls._default_sinks = sinks

        return cls

    return decorator


def logger_name(name: str) -> callable:
    """Decorator to set a default name for a logger class.

    This decorator allows you to specify a default name that will be used
    when no explicit name is provided to the logger constructor.

    Args:
        name: Default name to use for instances of this logger class

    Returns:
        A decorator function that configures the class with the specified name

    Example:
        @logger_name("MLTraining")
        class MyLogger(LoggerBase):
            loss: MeanField

        logger = MyLogger()  # Uses name "MLTraining"
        logger = MyLogger(name="Custom")  # Overrides with "Custom"
    """

    def decorator(cls: Type[T]) -> Type[T]:
        """The actual decorator function.

        Args:
            cls: The logger class to decorate

        Returns:
            The same class with _default_name attribute added
        """
        if not isinstance(name, str):
            raise TypeError("logger_name() expects a string")

        if not name.strip():
            raise ValueError("logger_name() expects a non-empty string")

        # Attach default name to the class
        cls._default_name = name.strip()

        return cls

    return decorator
