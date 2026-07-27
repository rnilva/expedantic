"""Regression tests for the four logger gotchas found while integrating the
``expedantic.logger`` into a real RL training loop.

1. ``from __future__ import annotations`` must not break ``LoggerBase`` schema resolution.
2. A public ``LoggerBase.add_field`` for runtime/dynamic fields.
3. ``FileSink`` JSONL and ``to_dataframe()`` must have identical keys (``_logger`` consistency).
4. ``WandBSink`` documents multiple sinks sharing a step (docstring only — asserted lightly).
"""

import json
import tempfile
from pathlib import Path

import pytest

from expedantic.logger import (
    LoggerBase,
    Field,
    MeanField,
    MaxField,
    FileSink,
    WandBSink,
)


# ---------------------------------------------------------------------------
# Gotcha 1: from __future__ import annotations
# ---------------------------------------------------------------------------
class TestFutureAnnotations:
    """A subclass that uses PEP 563 (stringized annotations) must construct and work."""

    def test_construct_and_log_with_future_annotations(self):
        # Imported lazily so the helper module's __future__ import is exercised on use.
        # pytest (rootdir-insert import mode) puts the tests/ dir on sys.path, so the
        # sibling helper module is importable by its bare name.
        from _future_annotations_logger import FutureAnnotatedLogger

        logger = FutureAnnotatedLogger(sinks=[])

        # Schema resolved despite stringized annotations.
        assert set(logger.schema) == {"iteration", "loss", "best"}
        assert isinstance(logger.iteration, Field)
        assert isinstance(logger.loss, MeanField)
        assert isinstance(logger.best, MaxField)

        # Logging and flushing works end to end.
        logger.iteration.log(7)
        logger.loss.log(1.0)
        logger.loss.log(3.0)
        logger.best.log(0.5)
        logger.best.log(0.9)

        row = logger.flush()
        assert row["iteration"] == 7
        assert row["loss"] == 2.0
        assert row["best"] == 0.9


# ---------------------------------------------------------------------------
# Gotcha 2: public add_field for dynamic fields
# ---------------------------------------------------------------------------
class TestAddField:
    """LoggerBase.add_field registers fields not known at class-definition time."""

    def _make_logger(self):
        class DynLogger(LoggerBase):
            step: Field[int]

        return DynLogger(sinks=[])

    def test_add_field_registers_logs_and_flushes(self):
        logger = self._make_logger()

        field = logger.add_field("extra", Field)
        assert "extra" in logger.schema
        assert isinstance(logger.extra, Field)
        assert field is logger.extra

        logger.step.log(3)
        logger.extra.log(42)
        row = logger.flush()

        assert row["step"] == 3
        assert row["extra"] == 42

    def test_add_field_parameterized_and_aggregating(self):
        logger = self._make_logger()
        logger.add_field("avg", MeanField)
        logger.add_field("typed", Field[int])

        logger.avg.log(2.0)
        logger.avg.log(4.0)
        logger.typed.log(11)
        row = logger.flush()

        assert row["avg"] == 3.0
        assert row["typed"] == 11

    def test_add_field_idempotent_same_type(self):
        logger = self._make_logger()
        f1 = logger.add_field("x", Field)
        logger.x.log(99)
        # Re-adding the same type must NOT reset the field.
        f2 = logger.add_field("x", Field)
        assert f1 is f2
        assert logger.x.value == 99

    def test_add_field_conflict_raises(self):
        logger = self._make_logger()
        logger.add_field("x", Field)
        with pytest.raises(ValueError, match="already registered"):
            logger.add_field("x", MeanField)

    def test_add_field_rejects_non_field(self):
        logger = self._make_logger()
        with pytest.raises(TypeError):
            logger.add_field("bad", int)


# ---------------------------------------------------------------------------
# Gotcha 3: FileSink JSONL and to_dataframe() agree on keys
# ---------------------------------------------------------------------------
class TestLoggerColumnConsistency:
    """The JSONL row (FileSink) and the in-memory data / dataframe must have the
    SAME keys — in particular ``_logger`` must appear in both or neither."""

    def test_filesink_and_dataframe_have_same_keys(self):
        class ConsistencyLogger(LoggerBase):
            step: Field[int]
            loss: MeanField

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "log.jsonl"
            logger = ConsistencyLogger(name="MyRun", sinks=[FileSink(path)])

            logger.step.log(1)
            logger.loss.log(0.25)
            logger.flush()
            logger.close()

            # Keys from the JSONL line.
            with open(path) as fh:
                jsonl_row = json.loads(fh.readline())
            jsonl_keys = set(jsonl_row)

            # Keys from to_dataframe().
            df = logger.to_dataframe()
            df_keys = set(df.columns)

            assert jsonl_keys == df_keys
            # And the logger name is present identically in both.
            assert "_logger" in jsonl_keys
            assert jsonl_row["_logger"] == "MyRun"
            assert df["_logger"].to_list() == ["MyRun"]

    def test_logger_name_in_canonical_data(self):
        class C(LoggerBase):
            step: Field[int]

        logger = C(name="run42", sinks=[])
        logger.step.log(5)
        row = logger.flush()
        assert row["_logger"] == "run42"
        assert logger.data[-1]["_logger"] == "run42"


# ---------------------------------------------------------------------------
# Gotcha 4: multiple WandBSink sharing a step (docstring + benign behavior)
# ---------------------------------------------------------------------------
class TestWandBSharedStep:
    def test_wandb_docstring_mentions_shared_step(self):
        assert "share" in WandBSink.__doc__.lower()
        assert "step" in WandBSink.__doc__.lower()
