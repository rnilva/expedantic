"""Helper module for the gotcha-1 test.

This module deliberately enables ``from __future__ import annotations`` so that all
annotations below are stored as *strings*. A ``LoggerBase`` subclass defined here
must still construct correctly: the schema resolution has to evaluate the stringized
``Field[int]`` / ``MeanField`` annotations against THIS module's globals (where the
field types are imported), not against some unrelated namespace.

Kept in its own module on purpose: ``from __future__ import annotations`` is
file-scoped, and we don't want it to leak into the rest of the test suite.
"""

from __future__ import annotations

from expedantic.logger import Field, MeanField, MaxField, LoggerBase


class FutureAnnotatedLogger(LoggerBase):
    """A logger whose annotations are stringized by PEP 563."""

    iteration: Field[int]
    loss: MeanField
    best: MaxField[float]
