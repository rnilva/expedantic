"""Example logger implementations.

This module contains pre-built logger examples that demonstrate
common patterns and can be used as starting points.
"""

from .base import LoggerBase
from .fields import Field, MeanField, MaxField, SumField, ListField


class TrainingLogger(LoggerBase):
    """Example logger for machine learning training loops.

    This logger demonstrates common patterns for tracking training metrics:
    - Current state values (iteration, epoch, learning_rate)
    - Averaged metrics (loss, val_loss)
    - Peak performance tracking (accuracy)
    - Accumulated values (batch_time)
    - Event collection (messages)
    """

    iteration: Field[int]  # Current training iteration
    epoch: Field[int]  # Current epoch number
    loss: MeanField  # Average training loss per epoch
    val_loss: MeanField  # Average validation loss per epoch
    learning_rate: Field[float]  # Current learning rate
    messages: ListField[str]  # Collected log messages
    accuracy: MaxField[float]  # Best accuracy achieved in epoch
    batch_time: SumField[float]  # Total time spent on batches
