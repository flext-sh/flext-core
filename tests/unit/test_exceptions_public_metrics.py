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
from flext_tests import e

from tests.models import m


class TestsFlextCoreExceptionsPublicMetrics:
    """Public-contract behavior of exception occurrence metrics."""

    pytestmark = pytest.mark.usefixtures("_isolated_metrics")

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
        assert isinstance(snapshot, m.ExceptionMetricsSnapshot)

    @staticmethod
    def test_cleared_state_reports_no_exceptions() -> None:
        # Act
        """Test cleared state reports no exceptions."""
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        assert snapshot.total_exceptions == 0
        assert snapshot.unique_exception_types == 0
        assert not snapshot.exception_counts_summary
        assert snapshot.has_exceptions is False
        assert dict(snapshot.exception_counts) == {}

    @staticmethod
    def test_recording_one_exception_makes_metrics_non_empty() -> None:
        # Act
        """Test recording one exception makes metrics non empty."""
        e.record_exception(e.ValidationError)
        snapshot = e.resolve_metrics_snapshot()

        # Assert
        assert snapshot.total_exceptions == 1
        assert snapshot.unique_exception_types == 1
        assert snapshot.has_exceptions is True
        assert snapshot.exception_counts[e.ValidationError.__qualname__] == 1

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
        assert snapshot.total_exceptions == expected_total
        assert snapshot.unique_exception_types == expected_unique

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
        assert snapshot.exception_counts[e.ValidationError.__qualname__] == 2
        assert snapshot.exception_counts[e.FlextTimeoutError.__qualname__] == 3

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
        assert "ValidationError:2" in summary
        assert "TimeoutError:1" in summary

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
        assert snapshot.total_exceptions == 0
        assert snapshot.unique_exception_types == 0
        assert not snapshot.exception_counts_summary
        assert snapshot.has_exceptions is False

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
        assert snapshot.total_exceptions == 0

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
        assert first.total_exceptions == 1
        assert second.total_exceptions == 2

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
        assert config["total_exceptions"] == 3
        assert config["unique_exception_types"] == 2
        assert config[f"exception_counts.{e.ValidationError.__qualname__}"] == 2
        assert config[f"exception_counts.{e.FlextTimeoutError.__qualname__}"] == 1
