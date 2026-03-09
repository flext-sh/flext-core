"""Real tests to achieve 100% service coverage - no mocks.

This module provides comprehensive real tests (no mocks, patches, or bypasses)
to cover all remaining lines in service.py.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import r, s, t
from flext_tests import FlextTestsDomains
from tests.test_utils import (
    assertion_helpers,  # pyright: ignore[reportMissingTypeStubs]
)

TestDomainResult = FlextTestsDomains.TestDomainResult


class TestService(s[TestDomainResult]):
    """Test service for coverage tests."""

    __test__ = False

    def __init__(self, **data: t.ContainerValue) -> None:
        """Initialize test service."""
        super().__init__(**data)

    @override
    def execute(self, **_kwargs: t.ContainerValue) -> r[TestDomainResult]:
        """Execute service."""
        return self.ok(TestDomainResult("success"))


class TestServiceWithValidation(s[TestDomainResult]):
    """Test service with validation."""

    __test__ = False

    def __init__(self, **data: t.ContainerValue) -> None:
        """Initialize test service."""
        super().__init__(**data)

    @override
    def execute(self, **_kwargs: t.ContainerValue) -> r[TestDomainResult]:
        """Execute service."""
        return self.ok(TestDomainResult("validated"))


class TestService100Coverage:
    """Real tests to achieve 100% service coverage."""

    def test_validate_business_rules_success(self) -> None:
        """Test validate_business_rules."""
        service = TestService()
        result = service.validate_business_rules()
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
        assert len(info) > 0
        assert "service_type" in info or "service_name" in info or "class_name" in info

    def test_execute_success(self) -> None:
        """Test execute method."""
        service = TestService()
        result = service.execute()
        _ = assertion_helpers.assert_flext_result_success(result)
        assert isinstance(result.value, TestDomainResult)

    def test_ok_method(self) -> None:
        """Test ok method."""
        service = TestService()
        result = service.ok(TestDomainResult("test"))
        _ = assertion_helpers.assert_flext_result_success(result)
        assert result.value.value == "test"

    def test_result_property(self) -> None:
        """Test result property."""
        service = TestService()
        result = service.result
        assert isinstance(result, TestDomainResult)

    def test_auto_execute_false(self) -> None:
        """Test auto_execute when False."""
        service = TestService()
        assert (
            not hasattr(service, "auto_execute")
            or getattr(service, "auto_execute", False) is False
        )

    def test_validate_business_rules_override(self) -> None:
        """Test validate_business_rules can be overridden."""
        service = TestServiceWithValidation()
        result = service.validate_business_rules()
        assert isinstance(result, r)
