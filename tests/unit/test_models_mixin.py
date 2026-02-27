"""Tests for FlextModels.Mixin namespace models.

Module: flext_core._models.mixin
Scope: Mixin state models — OperationStats validation, computed fields,
and m.Mixin.* namespace accessibility.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest
from flext_core.models import m
from pydantic import ValidationError


class TestMixinNamespace:
    """Verify m.Mixin.* namespace is accessible."""

    def test_mixin_namespace_exists(self) -> None:
        """m.Mixin must be accessible from the facade."""
        assert hasattr(m, "Mixin")

    def test_operation_stats_accessible(self) -> None:
        """m.Mixin.OperationStats must be accessible."""
        assert hasattr(m.Mixin, "OperationStats")

    def test_service_runtime_not_duplicated(self) -> None:
        """m.ServiceRuntime must remain the canonical runtime triple model."""
        assert hasattr(m, "ServiceRuntime")
        # Mixin namespace should NOT duplicate ServiceRuntime
        assert not hasattr(m.Mixin, "ServiceRuntime")


class TestOperationStats:
    """Validate m.Mixin.OperationStats model behavior."""

    def test_default_construction(self) -> None:
        """Default OperationStats should have zero values."""
        stats = m.Mixin.OperationStats()
        assert stats.operation_count == 0
        assert stats.error_count == 0
        assert stats.total_duration_ms == 0.0
        assert stats.avg_duration_ms == 0.0
        assert stats.success_rate == 0.0

    def test_success_rate_computed(self) -> None:
        """success_rate should be computed from operation_count and error_count."""
        stats = m.Mixin.OperationStats(operation_count=10, error_count=2)
        assert stats.success_rate == pytest.approx(0.8)

    def test_success_rate_zero_operations(self) -> None:
        """success_rate with zero operations should be 0.0, not raise ZeroDivisionError."""
        stats = m.Mixin.OperationStats()
        assert stats.success_rate == 0.0

    def test_success_rate_no_errors(self) -> None:
        """success_rate with no errors should be 1.0."""
        stats = m.Mixin.OperationStats(operation_count=5)
        assert stats.success_rate == pytest.approx(1.0)

    def test_success_rate_all_errors(self) -> None:
        """success_rate when all operations fail should be 0.0."""
        stats = m.Mixin.OperationStats(operation_count=5, error_count=5)
        assert stats.success_rate == pytest.approx(0.0)

    def test_model_dump_includes_computed(self) -> None:
        """model_dump should include the computed success_rate field."""
        stats = m.Mixin.OperationStats(operation_count=4, error_count=1)
        dump = stats.model_dump()
        assert "success_rate" in dump
        assert dump["success_rate"] == pytest.approx(0.75)

    def test_model_dump_json(self) -> None:
        """model_dump_json should produce valid JSON with all fields."""
        stats = m.Mixin.OperationStats(
            operation_count=10,
            error_count=3,
            total_duration_ms=500.0,
            avg_duration_ms=50.0,
        )
        json_str = stats.model_dump_json()
        assert '"operation_count":10' in json_str
        assert '"success_rate":' in json_str

    def test_negative_operation_count_rejected(self) -> None:
        """Negative operation_count must be rejected by ge=0 constraint."""
        with pytest.raises(ValidationError):
            m.Mixin.OperationStats(operation_count=-1)

    def test_negative_error_count_rejected(self) -> None:
        """Negative error_count must be rejected by ge=0 constraint."""
        with pytest.raises(ValidationError):
            m.Mixin.OperationStats(error_count=-1)

    def test_negative_duration_rejected(self) -> None:
        """Negative total_duration_ms must be rejected by ge=0.0 constraint."""
        with pytest.raises(ValidationError):
            m.Mixin.OperationStats(total_duration_ms=-1.0)

    def test_validate_assignment(self) -> None:
        """validate_assignment=True should validate on field mutation."""
        stats = m.Mixin.OperationStats(operation_count=5)
        stats.operation_count = 10
        assert stats.operation_count == 10
        assert stats.success_rate == pytest.approx(1.0)

    def test_model_validate_from_dict(self) -> None:
        """model_validate should construct from dict (Pydantic v2 API)."""
        stats = m.Mixin.OperationStats.model_validate({
            "operation_count": 8,
            "error_count": 1,
            "total_duration_ms": 200.0,
            "avg_duration_ms": 25.0,
        })
        assert stats.operation_count == 8
        assert stats.success_rate == pytest.approx(0.875)
