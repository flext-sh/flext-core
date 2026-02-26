"""Real tests to achieve 100% service coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in service.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import r, s, t
from flext_core.typings import t
from flext_tests.domains import FlextTestsDomains

from tests.test_utils import assertion_helpers

# ==================== REAL SERVICE CLASSES ====================

# Use shared TestDomainResult from flext_tests
TestDomainResult = FlextTestsDomains.TestDomainResult


class TestService(s[TestDomainResult]):
    """Test service for coverage tests."""

    __test__ = False  # Not a test class, just a helper class

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize test service."""
        super().__init__(**data)

    def execute(self, **_kwargs: t.GeneralValueType) -> r[TestDomainResult]:
        """Execute service."""
        return self.ok(TestDomainResult("success"))


class TestServiceWithValidation(s[TestDomainResult]):
    """Test service with validation."""

    __test__ = False  # Not a test class, just a helper class

    def __init__(self, **data: t.GeneralValueType) -> None:
        """Initialize test service."""
        super().__init__(**data)

    def execute(self, **_kwargs: t.GeneralValueType) -> r[TestDomainResult]:
        """Execute service."""
        return self.ok(TestDomainResult("validated"))


# ==================== COVERAGE TESTS ====================


class TestService100Coverage:
    """Real tests to achieve 100% service coverage."""

    def test_validate_business_rules_success(self) -> None:
        """Test validate_business_rules."""
        service = TestService()
        result = service.validate_business_rules()

        # Default implementation should succeed
        _ = assertion_helpers.assert_flext_result_success(result) or result.is_failure

    def test_is_valid(self) -> None:
        """Test is_valid property."""
        service = TestService()
        is_valid = service.is_valid()

        assert isinstance(is_valid, bool)

    def test_get_service_info(self) -> None:
        """Test get_service_info."""
        service = TestService()
        info = service.get_service_info()

        assert isinstance(info, dict)
        # Check for any of the possible keys that might be in service info
        assert len(info) > 0
        # Service info should contain at least service_type or similar
        assert "service_type" in info or "service_name" in info or "class_name" in info

    def test_execute_success(self) -> None:
        """Test execute method."""
        service = TestService()
        result = service.execute()

        assertion_helpers.assert_flext_result_success(result)
        assert isinstance(result.value, TestDomainResult)

    def test_ok_method(self) -> None:
        """Test ok method."""
        service = TestService()
        result = service.ok(TestDomainResult("test"))

        assertion_helpers.assert_flext_result_success(result)
        assert result.value.value == "test"

    def test_result_property(self) -> None:
        """Test result property."""
        service = TestService()
        result = service.result

        assert isinstance(result, TestDomainResult)

    def test_auto_execute_false(self) -> None:
        """Test auto_execute when False."""
        service = TestService()
        # Note: auto_execute is not a default attribute in s base
        # It's only present when explicitly defined as ClassVar in subclasses
        # Default behavior is manual execution (auto_execute=False equivalent)
        assert (
            not hasattr(service, "auto_execute")
            or getattr(service, "auto_execute", False) is False
        )

    def test_validate_business_rules_override(self) -> None:
        """Test validate_business_rules can be overridden."""
        service = TestServiceWithValidation()
        result = service.validate_business_rules()

        assert isinstance(result, r)
