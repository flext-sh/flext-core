"""Coverage boost tests for uncovered lines.

This module contains targeted tests to increase code coverage
for specific uncovered lines in the codebase, focusing on critical modules
that need to reach 95%+ coverage.

Target Modules:
    - container.py: 90% → 95%+
    - foundation.py: 75% → 95%+
    - exceptions.py: 86% → 95%+
    - models.py: 71% → 95%+
    - payload.py: 80% → 95%+
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from flext_core.__version__ import get_version_info
from flext_core.container import FlextContainer
from flext_core.exceptions import FlextError
from flext_core.models import FlextFactory
from flext_core.payload import FlextPayload
from flext_core.result import FlextResult
from flext_core.utilities import flext_safe_int_conversion
from flext_core.value_objects import FlextValueObject

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestCoverageBoost:
    """Tests to boost coverage for specific uncovered lines."""

    def test_safe_int_conversion_coverage(self) -> None:
        """Test safe int conversion function edge cases."""
        # Test edge cases to increase coverage
        result = flext_safe_int_conversion("not_a_number", 42)
        assert result == 42

        result = flext_safe_int_conversion(None, None)
        assert result is None

    def test_version_info_coverage(self) -> None:
        """Test version info function."""
        # Simple test to increase coverage
        version_info = get_version_info()
        assert version_info.major >= 0
        assert version_info.minor >= 0
        assert version_info.patch >= 0

    def test_container_edge_cases(self) -> None:
        """Test container edge cases and error paths."""
        container = FlextContainer()

        # Test invalid service names to boost container coverage
        result = container.register("", Mock())
        assert result.is_failure
        assert "empty" in str(result.error).lower()

        # Test non-callable factory registration
        result = container.register_factory("test_service", "not_callable")
        assert result.is_failure

    def test_exception_edge_cases(self) -> None:
        """Test exception edge cases for coverage."""

        # Test FlextError with sensitive data redaction
        error = FlextError(
            "Security error",
            context={"password": "secret", "api_key": "key123", "normal": "data"},
        )
        error_dict = error.to_dict()

        # Verify sensitive fields are redacted
        assert error_dict["context"]["password"] == "[REDACTED]"
        assert error_dict["context"]["api_key"] == "[REDACTED]"
        assert error_dict["context"]["normal"] == "data"

    def test_payload_edge_cases(self) -> None:
        """Test payload edge cases for coverage boost."""

        # Test FlextPayload with complex data
        complex_data = {"nested": {"list": [1, 2, 3]}}
        result = FlextPayload.create(complex_data, source="test")
        assert result.is_success

        payload = result.unwrap()

        # Test get_data_or_default with default
        default = {"default": "value"}
        actual = payload.get_data_or_default(default)
        assert actual == complex_data

    def test_factory_error_paths(self) -> None:
        """Test FlextFactory error handling paths."""

        class TestVO(FlextValueObject):
            name: str

            def validate_business_rules(self) -> FlextResult[None]:
                if not self.name:
                    return FlextResult.fail("Name cannot be empty")
                return FlextResult.ok(None)

        # Test business rule validation failure
        result = FlextFactory.create_model(TestVO, name="")
        assert result.is_failure
        assert "Name cannot be empty" in str(result.error)
