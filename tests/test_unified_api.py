"""Test unified API usage patterns for the new consolidated classes.

This test demonstrates correct usage of the unified API after consolidation:
- Import from root module only (from flext_core import X)
- Use consolidated classes (FlextModels.Payload instead of FlextModels.Payload)
- Use unified API patterns (FlextResult, FlextContainer, etc.)

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import pytest

# ✅ CORRECT: Import from root module only
from flext_core import (
    FlextConstants,
    FlextContainer,
    FlextModels,
    FlextResult,
    FlextTypes,
    FlextValidation,
    get_flext_container,
)

pytestmark = [pytest.mark.unit, pytest.mark.core]


class TestUnifiedAPIPatterns:
    """Test unified API usage patterns."""

    def test_basic_imports_work(self):
        """Test that all unified imports work."""
        # All imports should work without errors
        assert FlextResult is not None
        assert FlextContainer is not None
        assert FlextModels is not None
        assert FlextConstants is not None
        assert FlextTypes is not None
        assert get_flext_container is not None

    def test_flext_result_unified_api(self):
        """Test FlextResult unified API patterns."""
        # ✅ CORRECT: Use FlextResult unified API
        success = FlextResult.ok("test_data")
        assert success.success
        assert success.unwrap() == "test_data"

        failure = FlextResult.fail("test_error")
        assert failure.failure
        assert failure.error == "test_error"

        # Test railway-oriented programming
        result = (
            FlextResult.ok(5)
            .map(lambda x: x * 2)
            .flat_map(lambda x: FlextResult.ok(x + 1))
        )
        assert result.success
        assert result.unwrap() == 11

    def test_flext_container_unified_api(self):
        """Test FlextContainer unified API patterns."""
        # ✅ CORRECT: Use global container function
        container = get_flext_container()

        # Register and retrieve services
        container.register("test_service", {"config": "test"})

        result = container.get("test_service")
        assert result.success

        service = result.unwrap()
        assert service["config"] == "test"

    def test_consolidated_models_api(self):
        """Test consolidated FlextModels API patterns."""
        # ✅ CORRECT: Use FlextModels.Payload instead of FlextModels.Payload
        assert hasattr(FlextModels, "Payload")

        # Test payload creation through FlextModels
        payload_data = {"test": "data"}
        result = FlextModels.create_payload(
            payload_data, message_type="test_message", source="test_service"
        )

        assert result.success
        payload = result.unwrap()
        assert payload.data == payload_data

    def test_consolidated_constants_api(self):
        """Test consolidated FlextConstants API patterns."""
        # ✅ CORRECT: Use FlextConstants.Config instead of separate imports
        assert hasattr(FlextConstants, "Config")
        assert hasattr(FlextConstants.Config, "ConfigEnvironment")
        assert hasattr(FlextConstants.Config, "LogLevel")

        # Test enum access
        env = FlextConstants.Config.ConfigEnvironment.DEVELOPMENT
        assert env.value == "development"

    def test_consolidated_types_api(self):
        """Test consolidated FlextTypes API patterns."""
        # ✅ CORRECT: Use FlextTypes.Core instead of separate imports
        assert hasattr(FlextTypes, "Core")
        assert hasattr(FlextTypes, "Config")
        assert hasattr(FlextTypes, "Domain")

        # Types should be accessible hierarchically
        assert hasattr(FlextTypes.Core, "JsonObject")
        assert hasattr(FlextTypes.Config, "ConfigDict")

    def test_validation_unified_api(self):
        """Test FlextValidation unified API patterns."""
        # ✅ CORRECT: Use FlextValidation for all validation needs
        assert hasattr(FlextValidation, "validate_email")
        assert hasattr(FlextValidation, "validate_required")

        # Test email validation
        result = FlextValidation.validate_email("test@example.com")
        assert result.success

        # Test required field validation
        result = FlextValidation.validate_required({"name": "John"}, "name")
        assert result.success

    def test_error_handling_patterns(self):
        """Test unified error handling patterns."""

        # ✅ CORRECT: Always use FlextResult for error handling
        def risky_operation(value: int) -> FlextResult[int]:
            if value < 0:
                return FlextResult.fail("Negative values not allowed")
            return FlextResult.ok(value * 2)

        # Test success case
        success_result = risky_operation(5)
        assert success_result.success
        assert success_result.unwrap() == 10

        # Test error case
        error_result = risky_operation(-1)
        assert error_result.failure
        assert "Negative values not allowed" in error_result.error

    def test_service_integration_patterns(self):
        """Test service integration with unified API."""
        # ✅ CORRECT: Service integration through container
        container = get_flext_container()

        # Register factory function
        def create_service():
            return {"initialized": True, "status": "ready"}

        container.register_factory("dynamic_service", create_service)

        # Retrieve and test service
        result = container.get("dynamic_service")
        assert result.success

        service = result.unwrap()
        assert service["initialized"] is True
        assert service["status"] == "ready"
