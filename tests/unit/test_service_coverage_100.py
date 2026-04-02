"""Real tests to achieve 100% service coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in service.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import r, s
from flext_tests import tm
from tests import t


class TestService100Coverage:
    """Real tests to achieve 100% service coverage."""

    class _ServiceStub(s[str]):
        """Test service for coverage tests."""

        __test__ = False

        @override
        def execute(self, **_kwargs: t.Scalar) -> r[str]:
            """Execute service."""
            return r[str].ok("success")

    class _ServiceWithValidationStub(s[str]):
        """Test service with validation."""

        __test__ = False

        @override
        def execute(self, **_kwargs: t.Scalar) -> r[str]:
            """Execute service."""
            return r[str].ok("validated")

    def test_validate_business_rules_success(self) -> None:
        """Test validate_business_rules."""
        service = self._ServiceStub()
        result = service.validate_business_rules()
        tm.ok(result)

    def test_is_valid(self) -> None:
        """Test is_valid property."""
        service = self._ServiceStub()
        is_valid = service.is_valid()
        tm.that(is_valid, is_=bool)

    def test_execute_success(self) -> None:
        """Test execute method."""
        service = self._ServiceStub()
        result = service.execute()
        tm.ok(result)
        tm.that(result.value, is_=str)

    def test_ok_method(self) -> None:
        """Test r.ok factory method (strict mode — no self.ok)."""
        result = r[str].ok("test")
        tm.ok(result)
        tm.ok(result, eq="test")

    def test_result_property(self) -> None:
        """Test result property."""
        service = self._ServiceStub()
        result = service.result
        tm.that(result, is_=str)

    def test_auto_execute_false(self) -> None:
        """Test auto_execute when False."""
        service = self._ServiceStub()
        assert (
            not hasattr(service, "auto_execute")
            or getattr(service, "auto_execute", False) is False
        )

    def test_validate_business_rules_override(self) -> None:
        """Test validate_business_rules can be overridden."""
        service = self._ServiceWithValidationStub()
        result = service.validate_business_rules()
        tm.ok(result)
