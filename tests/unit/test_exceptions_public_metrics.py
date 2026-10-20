"""Behavioral tests for the public FlextExceptions metrics contract.

Exercises only the public surface: ``record_exception`` / ``clear_metrics`` /
``resolve_metrics_snapshot`` / ``resolve_metrics`` and the observable state of
the returned ``ExceptionMetricsSnapshot`` model. No private attributes, no
patched internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import pytest

from tests import e, m, tm


class TestsFlextCoreExceptionsPublicMetrics:
    """Public-contract behavior of exception occurrence metrics."""

    pytestmark = pytest.mark.usefixtures("_isolated_metrics")

    @staticmethod
    def test_model_defaults_are_independent_and_copy_updated() -> None:
        """Default validation keeps independent public metrics values."""
        first = m.ExceptionMetricsState()
        second = m.ExceptionMetricsState()
        snapshot = m.ExceptionMetricsSnapshot()

        tm.that(first.exception_counts, eq={})
        tm.that(first.exception_counts is not second.exception_counts, truth=True)
        tm.that(snapshot.exception_counts is not first.exception_counts, truth=True)
        updated = first.record_exception(ValueError)
        tm.that(updated.exception_counts, eq={ValueError.__qualname__: 1})
        tm.that(first.exception_counts, eq={})

    @staticmethod
    def test_invalid_metrics_preserve_native_validation_error() -> None:
        """Invalid public input must escape as the original Pydantic error."""
        with pytest.raises(m.ValidationError) as caught:
            m.ExceptionMetricsState.model_validate({"exception_counts": {"bad": "x"}})

        tm.that(caught.value.errors()[0]["loc"], eq=("exception_counts", "bad"))
        tm.that(caught.value.errors()[0]["type"], eq="int_type")

    @staticmethod
    @pytest.fixture
    def _isolated_metrics() -> Iterator[None]:
        """Guarantee each test observes a clean, isolated metrics state."""
        e.clear_metrics()
        yield
        e.clear_metrics()

    @staticmethod
    def test_snapshot_is_the_public_model_type() -> None:
        # Act
        """Test snapshot is the public model type."""
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot, is_=m.ExceptionMetricsSnapshot)

    @staticmethod
    def test_cleared_state_reports_no_exceptions() -> None:
        # Act
        """Test cleared state reports no exceptions."""
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.total_exceptions, eq=0)
        tm.that(snapshot.unique_exception_types, eq=0)
        tm.that(snapshot.exception_counts_summary, truth=False)
        tm.that(snapshot.has_exceptions is False, truth=True)
        tm.that(dict(snapshot.exception_counts), eq={})

    @staticmethod
    def test_recording_one_exception_makes_metrics_non_empty() -> None:
        # Act
        """Test recording one exception makes metrics non empty."""
        e.record_exception(e.ValidationError)
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.total_exceptions, eq=1)
        tm.that(snapshot.unique_exception_types, eq=1)
        tm.that(snapshot.has_exceptions is True, truth=True)
        tm.that(snapshot.exception_counts[e.ValidationError.__qualname__], eq=1)

    @pytest.mark.parametrize(
        ("recorded", "expected_total", "expected_unique"),
        [
            ((e.ValidationError,), 1, 1),
            ((e.ValidationError, e.ValidationError), 2, 1),
            ((e.ValidationError, e.FlextTimeoutError), 2, 2),
            (
                (
                    e.ValidationError,
                    e.ValidationError,
                    e.FlextTimeoutError,
                    e.FlextTimeoutError,
                    e.FlextTimeoutError,
                ),
                5,
                2,
            ),
        ],
    )
    @staticmethod
    def test_totals_and_unique_counts_track_recorded_exceptions(
        recorded: Sequence[type[BaseException]],
        expected_total: int,
        expected_unique: int,
    ) -> None:
        # Act
        """Test totals and unique counts track recorded exceptions."""
        for exception_type in recorded:
            e.record_exception(exception_type)
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.total_exceptions, eq=expected_total)
        tm.that(snapshot.unique_exception_types, eq=expected_unique)

    @staticmethod
    def test_per_type_counts_are_keyed_by_qualified_name() -> None:
        # Arrange
        """Test per type counts are keyed by qualified name."""
        for exception_type in (
            e.ValidationError,
            e.ValidationError,
            e.FlextTimeoutError,
            e.FlextTimeoutError,
            e.FlextTimeoutError,
        ):
            e.record_exception(exception_type)

        # Act
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.exception_counts[e.ValidationError.__qualname__], eq=2)
        tm.that(snapshot.exception_counts[e.FlextTimeoutError.__qualname__], eq=3)

    @staticmethod
    def test_summary_reports_name_and_count_for_each_type() -> None:
        # Arrange
        """Test summary reports name and count for each type."""
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.FlextTimeoutError)

        # Act
        summary = e.resolve_metrics_snapshot().exception_counts_summary

        # Assert
        tm.that("ValidationError:2" in summary, truth=True)
        tm.that("TimeoutError:1" in summary, truth=True)

    @staticmethod
    def test_clear_metrics_resets_all_public_totals() -> None:
        # Arrange
        """Test clear metrics resets all public totals."""
        e.record_exception(e.ValidationError)
        e.record_exception(e.FlextTimeoutError)

        # Act
        e.clear_metrics()
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.total_exceptions, eq=0)
        tm.that(snapshot.unique_exception_types, eq=0)
        tm.that(snapshot.exception_counts_summary, truth=False)
        tm.that(snapshot.has_exceptions is False, truth=True)

    @staticmethod
    def test_clear_metrics_is_idempotent() -> None:
        # Arrange
        """Test clear metrics is idempotent."""
        e.record_exception(e.ValidationError)

        # Act
        e.clear_metrics()
        e.clear_metrics()
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        tm.that(snapshot.total_exceptions, eq=0)

    @staticmethod
    def test_snapshot_is_a_stable_value_independent_of_later_recording() -> None:
        # Arrange
        """Test snapshot is a stable value independent of later recording."""
        e.record_exception(e.ValidationError)
        first = e.resolve_metrics_snapshot()

        # Act
        e.record_exception(e.ValidationError)
        second = e.resolve_metrics_snapshot()

        # Assert — the earlier snapshot is a value, not a live view
        tm.that(first.total_exceptions, eq=1)
        tm.that(second.total_exceptions, eq=2)

    @staticmethod
    def test_resolve_metrics_exposes_flat_config_contract() -> None:
        # Arrange
        """Test resolve metrics exposes flat config contract."""
        e.record_exception(e.ValidationError)
        e.record_exception(e.ValidationError)
        e.record_exception(e.FlextTimeoutError)

        # Act
        config = e.resolve_metrics()

        # Assert
        tm.that(config["total_exceptions"], eq=3)
        tm.that(config["unique_exception_types"], eq=2)
        tm.that(config[f"exception_counts.{e.ValidationError.__qualname__}"], eq=2)
        tm.that(config[f"exception_counts.{e.FlextTimeoutError.__qualname__}"], eq=1)
