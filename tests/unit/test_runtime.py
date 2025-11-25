"""Refactored comprehensive tests for FlextRuntime - Layer 0.5 Runtime Utilities.

Tests all functionality of FlextRuntime including type guards, serialization utilities,
external library access, type introspection, and structlog configuration.

Consolidates 6 test classes with 45+ methods into 1 unified class with
parametrized and integration tests using Python 3.13 patterns.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import dataclasses
import logging
from enum import StrEnum
from typing import ClassVar

import pytest
import structlog
from dependency_injector import containers, providers

from flext_core import (
    FlextConstants,
    FlextContext,
    FlextRuntime,
)

# =========================================================================
# Operation Type Enumeration
# =========================================================================


class RuntimeOperationType(StrEnum):
    """Runtime operation types for parametrized testing."""

    # Type guards
    PHONE_VALID = "phone_valid"
    PHONE_INVALID = "phone_invalid"
    PHONE_NON_STRING = "phone_non_string"
    DICT_LIKE_VALID = "dict_like_valid"
    DICT_LIKE_INVALID = "dict_like_invalid"
    LIST_LIKE_VALID = "list_like_valid"
    LIST_LIKE_INVALID = "list_like_invalid"
    JSON_VALID = "json_valid"
    JSON_INVALID = "json_invalid"
    JSON_NON_STRING = "json_non_string"
    IDENTIFIER_VALID = "identifier_valid"
    IDENTIFIER_INVALID = "identifier_invalid"
    IDENTIFIER_NON_STRING = "identifier_non_string"

    # Serialization
    SAFE_GET_ATTRIBUTE_EXISTS = "safe_get_attribute_exists"
    SAFE_GET_ATTRIBUTE_MISSING_DEFAULT = "safe_get_attribute_missing_default"
    SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT = "safe_get_attribute_missing_no_default"

    # Type introspection
    EXTRACT_GENERIC_GENERIC_TYPE = "extract_generic_generic_type"
    EXTRACT_GENERIC_NON_GENERIC = "extract_generic_non_generic"
    EXTRACT_GENERIC_EXCEPTION = "extract_generic_exception"
    SEQUENCE_TYPE_VALID = "sequence_type_valid"
    SEQUENCE_TYPE_INVALID = "sequence_type_invalid"
    SEQUENCE_TYPE_EXCEPTION = "sequence_type_exception"

    # External library access
    STRUCTLOG_MODULE = "structlog_module"
    DEPENDENCY_PROVIDERS_MODULE = "dependency_providers_module"
    DEPENDENCY_CONTAINERS_MODULE = "dependency_containers_module"

    # Structlog configuration
    CONFIGURE_STRUCTLOG_DEFAULTS = "configure_structlog_defaults"
    CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL = "configure_structlog_custom_log_level"
    CONFIGURE_STRUCTLOG_JSON_RENDERER = "configure_structlog_json_renderer"
    CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS = (
        "configure_structlog_additional_processors"
    )
    CONFIGURE_STRUCTLOG_IDEMPOTENT = "configure_structlog_idempotent"

    # Integration
    INTEGRATION_CONSTANTS_PATTERNS = "integration_constants_patterns"
    INTEGRATION_LAYER_HIERARCHY = "integration_layer_hierarchy"
    TRACK_SERVICE_RESOLUTION_SUCCESS = "track_service_resolution_success"
    TRACK_SERVICE_RESOLUTION_FAILURE = "track_service_resolution_failure"
    TRACK_DOMAIN_EVENT_WITH_AGGREGATE = "track_domain_event_with_aggregate"
    TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE = "track_domain_event_without_aggregate"
    SETUP_SERVICE_INFRASTRUCTURE_FULL = "setup_service_infrastructure_full"
    SETUP_SERVICE_INFRASTRUCTURE_MINIMAL = "setup_service_infrastructure_minimal"
    SETUP_SERVICE_WITHOUT_CORRELATION = "setup_service_without_correlation"


# =========================================================================
# Test Case Data Structure
# =========================================================================


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeTestCase:
    """Runtime test case definition with parametrization data."""

    name: str
    operation: RuntimeOperationType
    test_input: object = None
    expected_result: bool | tuple[object, ...] | object = None
    should_reset_config: bool = False


# =========================================================================
# Test Scenario Factory
# =========================================================================


class RuntimeScenarios:
    """Factory for runtime test scenarios with centralized test data."""

    # Type guard test cases
    PHONE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="phone_valid_international",
            operation=RuntimeOperationType.PHONE_VALID,
            test_input="+5511987654321",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="phone_valid_no_country",
            operation=RuntimeOperationType.PHONE_VALID,
            test_input="5511987654321",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="phone_valid_us_format",
            operation=RuntimeOperationType.PHONE_VALID,
            test_input="+1234567890",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="phone_valid_15_digits",
            operation=RuntimeOperationType.PHONE_VALID,
            test_input="123456789012345",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="phone_invalid_too_short",
            operation=RuntimeOperationType.PHONE_INVALID,
            test_input="123",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="phone_invalid_letters",
            operation=RuntimeOperationType.PHONE_INVALID,
            test_input="abc1234567890",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="phone_invalid_country_letters",
            operation=RuntimeOperationType.PHONE_INVALID,
            test_input="+abc123",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="phone_invalid_empty",
            operation=RuntimeOperationType.PHONE_INVALID,
            test_input="",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="phone_non_string_int",
            operation=RuntimeOperationType.PHONE_NON_STRING,
            test_input=5511987654321,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="phone_non_string_none",
            operation=RuntimeOperationType.PHONE_NON_STRING,
            test_input=None,
            expected_result=False,
        ),
    ]

    DICT_LIKE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="dict_like_empty",
            operation=RuntimeOperationType.DICT_LIKE_VALID,
            test_input={},
            expected_result=True,
        ),
        RuntimeTestCase(
            name="dict_like_single_key",
            operation=RuntimeOperationType.DICT_LIKE_VALID,
            test_input={"key": "value"},
            expected_result=True,
        ),
        RuntimeTestCase(
            name="dict_like_nested",
            operation=RuntimeOperationType.DICT_LIKE_VALID,
            test_input={"nested": {"dict": True}},
            expected_result=True,
        ),
        RuntimeTestCase(
            name="dict_like_invalid_list",
            operation=RuntimeOperationType.DICT_LIKE_INVALID,
            test_input=[],
            expected_result=False,
        ),
        RuntimeTestCase(
            name="dict_like_invalid_string",
            operation=RuntimeOperationType.DICT_LIKE_INVALID,
            test_input="string",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="dict_like_invalid_int",
            operation=RuntimeOperationType.DICT_LIKE_INVALID,
            test_input=123,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="dict_like_invalid_none",
            operation=RuntimeOperationType.DICT_LIKE_INVALID,
            test_input=None,
            expected_result=False,
        ),
    ]

    LIST_LIKE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="list_like_empty",
            operation=RuntimeOperationType.LIST_LIKE_VALID,
            test_input=[],
            expected_result=True,
        ),
        RuntimeTestCase(
            name="list_like_integers",
            operation=RuntimeOperationType.LIST_LIKE_VALID,
            test_input=[1, 2, 3],
            expected_result=True,
        ),
        RuntimeTestCase(
            name="list_like_strings",
            operation=RuntimeOperationType.LIST_LIKE_VALID,
            test_input=["a", "b", "c"],
            expected_result=True,
        ),
        RuntimeTestCase(
            name="list_like_invalid_dict",
            operation=RuntimeOperationType.LIST_LIKE_INVALID,
            test_input={},
            expected_result=False,
        ),
        RuntimeTestCase(
            name="list_like_invalid_string",
            operation=RuntimeOperationType.LIST_LIKE_INVALID,
            test_input="string",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="list_like_invalid_int",
            operation=RuntimeOperationType.LIST_LIKE_INVALID,
            test_input=123,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="list_like_invalid_none",
            operation=RuntimeOperationType.LIST_LIKE_INVALID,
            test_input=None,
            expected_result=False,
        ),
    ]

    JSON_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="json_valid_object",
            operation=RuntimeOperationType.JSON_VALID,
            test_input='{"key": "value"}',
            expected_result=True,
        ),
        RuntimeTestCase(
            name="json_valid_empty_array",
            operation=RuntimeOperationType.JSON_VALID,
            test_input="[]",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="json_valid_array",
            operation=RuntimeOperationType.JSON_VALID,
            test_input="[1, 2, 3]",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="json_valid_string",
            operation=RuntimeOperationType.JSON_VALID,
            test_input='"string"',
            expected_result=True,
        ),
        RuntimeTestCase(
            name="json_valid_null",
            operation=RuntimeOperationType.JSON_VALID,
            test_input="null",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="json_invalid_plain_text",
            operation=RuntimeOperationType.JSON_INVALID,
            test_input="not json",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="json_invalid_malformed",
            operation=RuntimeOperationType.JSON_INVALID,
            test_input="{invalid}",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="json_invalid_empty",
            operation=RuntimeOperationType.JSON_INVALID,
            test_input="",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="json_non_string_dict",
            operation=RuntimeOperationType.JSON_NON_STRING,
            test_input={"key": "value"},
            expected_result=False,
        ),
        RuntimeTestCase(
            name="json_non_string_list",
            operation=RuntimeOperationType.JSON_NON_STRING,
            test_input=[1, 2, 3],
            expected_result=False,
        ),
        RuntimeTestCase(
            name="json_non_string_none",
            operation=RuntimeOperationType.JSON_NON_STRING,
            test_input=None,
            expected_result=False,
        ),
    ]

    IDENTIFIER_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="identifier_valid_lowercase",
            operation=RuntimeOperationType.IDENTIFIER_VALID,
            test_input="variable",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="identifier_valid_private",
            operation=RuntimeOperationType.IDENTIFIER_VALID,
            test_input="_private",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="identifier_valid_class_name",
            operation=RuntimeOperationType.IDENTIFIER_VALID,
            test_input="ClassName",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="identifier_valid_snake_case",
            operation=RuntimeOperationType.IDENTIFIER_VALID,
            test_input="function_name",
            expected_result=True,
        ),
        RuntimeTestCase(
            name="identifier_invalid_starts_with_digit",
            operation=RuntimeOperationType.IDENTIFIER_INVALID,
            test_input="123invalid",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="identifier_invalid_hyphen",
            operation=RuntimeOperationType.IDENTIFIER_INVALID,
            test_input="invalid-name",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="identifier_invalid_space",
            operation=RuntimeOperationType.IDENTIFIER_INVALID,
            test_input="invalid name",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="identifier_invalid_empty",
            operation=RuntimeOperationType.IDENTIFIER_INVALID,
            test_input="",
            expected_result=False,
        ),
        RuntimeTestCase(
            name="identifier_non_string_int",
            operation=RuntimeOperationType.IDENTIFIER_NON_STRING,
            test_input=123,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="identifier_non_string_none",
            operation=RuntimeOperationType.IDENTIFIER_NON_STRING,
            test_input=None,
            expected_result=False,
        ),
    ]

    GENERIC_ARGS_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="extract_generic_list",
            operation=RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
            test_input=list[str],
            expected_result=(str,),
        ),
        RuntimeTestCase(
            name="extract_generic_dict",
            operation=RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
            test_input=dict[str, int],
            expected_result=(str, int),
        ),
        RuntimeTestCase(
            name="extract_generic_non_generic_str",
            operation=RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
            test_input=str,
            expected_result=(),
        ),
        RuntimeTestCase(
            name="extract_generic_non_generic_int",
            operation=RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
            test_input=int,
            expected_result=(),
        ),
        RuntimeTestCase(
            name="extract_generic_exception_none",
            operation=RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
            test_input=None,
            expected_result=(),
        ),
        RuntimeTestCase(
            name="extract_generic_exception_string",
            operation=RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
            test_input="not a type",
            expected_result=(),
        ),
    ]

    SEQUENCE_TYPE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="sequence_type_list_of_str",
            operation=RuntimeOperationType.SEQUENCE_TYPE_VALID,
            test_input=list[str],
            expected_result=True,
        ),
        RuntimeTestCase(
            name="sequence_type_tuple",
            operation=RuntimeOperationType.SEQUENCE_TYPE_VALID,
            test_input=tuple[int, ...],
            expected_result=True,
        ),
        RuntimeTestCase(
            name="sequence_type_str_is_sequence",
            operation=RuntimeOperationType.SEQUENCE_TYPE_VALID,
            test_input=str,
            expected_result=True,
        ),
        RuntimeTestCase(
            name="sequence_type_invalid_dict",
            operation=RuntimeOperationType.SEQUENCE_TYPE_INVALID,
            test_input=dict[str, int],
            expected_result=False,
        ),
        RuntimeTestCase(
            name="sequence_type_invalid_int",
            operation=RuntimeOperationType.SEQUENCE_TYPE_INVALID,
            test_input=int,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="sequence_type_exception_none",
            operation=RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
            test_input=None,
            expected_result=False,
        ),
        RuntimeTestCase(
            name="sequence_type_exception_string",
            operation=RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
            test_input="not a type",
            expected_result=False,
        ),
    ]

    SERIALIZATION_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="safe_get_attribute_exists",
            operation=RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS,
            expected_result="value",
        ),
        RuntimeTestCase(
            name="safe_get_attribute_missing_with_default",
            operation=RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT,
            expected_result="default",
        ),
        RuntimeTestCase(
            name="safe_get_attribute_missing_no_default",
            operation=RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT,
            expected_result=None,
        ),
    ]

    LIBRARY_ACCESS_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="structlog_module",
            operation=RuntimeOperationType.STRUCTLOG_MODULE,
            expected_result=structlog,
        ),
        RuntimeTestCase(
            name="dependency_providers",
            operation=RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE,
            expected_result=providers,
        ),
        RuntimeTestCase(
            name="dependency_containers",
            operation=RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE,
            expected_result=containers,
        ),
    ]

    STRUCTLOG_CONFIG_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="configure_defaults",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_custom_level",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_json_renderer",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_additional_processors",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_idempotent",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT,
            should_reset_config=True,
        ),
    ]

    INTEGRATION_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="constants_patterns",
            operation=RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS,
        ),
        RuntimeTestCase(
            name="layer_hierarchy",
            operation=RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY,
        ),
        RuntimeTestCase(
            name="track_service_resolution_success",
            operation=RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS,
        ),
        RuntimeTestCase(
            name="track_service_resolution_failure",
            operation=RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE,
        ),
        RuntimeTestCase(
            name="track_domain_event_with_aggregate",
            operation=RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE,
        ),
        RuntimeTestCase(
            name="track_domain_event_without_aggregate",
            operation=RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE,
        ),
        RuntimeTestCase(
            name="setup_service_infrastructure_full",
            operation=RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL,
        ),
        RuntimeTestCase(
            name="setup_service_infrastructure_minimal",
            operation=RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL,
        ),
        RuntimeTestCase(
            name="setup_service_without_correlation",
            operation=RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION,
        ),
    ]

    @staticmethod
    def reset_structlog_config() -> None:
        """Reset structlog configuration for testing."""
        FlextRuntime._structlog_configured = False


# =========================================================================
# Test Suite
# =========================================================================


class TestFlextRuntime:
    """Unified test suite for FlextRuntime with parametrized tests."""

    # =====================================================================
    # Type Guard Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.PHONE_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_phone_validation(self, test_case: RuntimeTestCase) -> None:
        """Test phone number validation."""
        result = FlextRuntime.is_valid_phone(test_case.test_input)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.DICT_LIKE_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_dict_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test dict-like object validation."""
        result = FlextRuntime.is_dict_like(test_case.test_input)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIST_LIKE_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_list_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test list-like object validation."""
        result = FlextRuntime.is_list_like(test_case.test_input)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.JSON_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_json_validation(self, test_case: RuntimeTestCase) -> None:
        """Test JSON string validation."""
        result = FlextRuntime.is_valid_json(test_case.test_input)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.IDENTIFIER_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_identifier_validation(self, test_case: RuntimeTestCase) -> None:
        """Test Python identifier validation."""
        result = FlextRuntime.is_valid_identifier(test_case.test_input)
        assert result == test_case.expected_result

    # =====================================================================
    # Serialization Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.SERIALIZATION_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_safe_get_attribute(self, test_case: RuntimeTestCase) -> None:
        """Test safe attribute retrieval."""
        if test_case.operation == RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS:

            class TestObj:
                attr = "value"

            obj = TestObj()
            result = FlextRuntime.safe_get_attribute(obj, "attr")
            assert result == "value"

        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT
        ):

            class TestObjDefault:
                pass

            result = FlextRuntime.safe_get_attribute(
                TestObjDefault(), "missing", "default"
            )
            assert result == "default"

        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT
        ):

            class TestObjNoDefault:
                pass

            result = FlextRuntime.safe_get_attribute(TestObjNoDefault(), "missing")
            assert result is None

    # =====================================================================
    # Type Introspection Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.GENERIC_ARGS_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_extract_generic_args(self, test_case: RuntimeTestCase) -> None:
        """Test extraction of generic type arguments."""
        args = FlextRuntime.extract_generic_args(test_case.test_input)
        assert args == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.SEQUENCE_TYPE_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_sequence_type_detection(self, test_case: RuntimeTestCase) -> None:
        """Test sequence type detection."""
        result = FlextRuntime.is_sequence_type(test_case.test_input)
        assert result == test_case.expected_result

    # =====================================================================
    # External Library Access Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIBRARY_ACCESS_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_external_library_access(self, test_case: RuntimeTestCase) -> None:
        """Test external library access."""
        if test_case.operation == RuntimeOperationType.STRUCTLOG_MODULE:
            module = FlextRuntime.structlog()
            assert module is structlog
            assert hasattr(module, "get_logger")
            assert hasattr(module, "configure")

        elif test_case.operation == RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE:
            module = FlextRuntime.dependency_providers()
            assert module is providers
            assert hasattr(module, "Singleton")
            assert hasattr(module, "Factory")

        elif test_case.operation == RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE:
            module = FlextRuntime.dependency_containers()
            assert module is containers
            assert hasattr(module, "DeclarativeContainer")

    # =====================================================================
    # Structlog Configuration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.STRUCTLOG_CONFIG_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_structlog_configuration(self, test_case: RuntimeTestCase) -> None:
        """Test structlog configuration."""
        if test_case.should_reset_config:
            RuntimeScenarios.reset_structlog_config()

        if test_case.operation == RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS:
            FlextRuntime.configure_structlog()
            assert FlextRuntime._structlog_configured is True
            assert structlog.is_configured()

        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL
        ):
            FlextRuntime.configure_structlog(log_level=logging.DEBUG)
            assert FlextRuntime._structlog_configured is True

        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER
        ):
            FlextRuntime.configure_structlog(console_renderer=False)
            assert FlextRuntime._structlog_configured is True

        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS
        ):

            def custom_processor(
                logger: object,
                method_name: str,
                event_dict: dict[str, object],
            ) -> dict[str, object]:
                event_dict["custom"] = True
                return event_dict

            FlextRuntime.configure_structlog(additional_processors=[custom_processor])
            assert FlextRuntime._structlog_configured is True

        elif test_case.operation == RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT:
            FlextRuntime.configure_structlog()
            FlextRuntime.configure_structlog()
            assert FlextRuntime._structlog_configured is True

    # =====================================================================
    # Integration Tests
    # =====================================================================

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.INTEGRATION_SCENARIOS,
        ids=lambda tc: tc.name,
    )
    def test_runtime_integration(self, test_case: RuntimeTestCase) -> None:
        """Test FlextRuntime integration scenarios."""
        if test_case.operation == RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS:
            assert hasattr(FlextConstants.Platform, "PATTERN_PHONE_NUMBER")
            test_phone = "+5511987654321"
            assert FlextRuntime.is_valid_phone(test_phone)

        elif test_case.operation == RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY:
            assert FlextConstants is not None
            assert FlextRuntime is not None
            assert FlextConstants.Platform.PATTERN_EMAIL is not None

        elif (
            test_case.operation == RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution("database", resolved=True)
            current_correlation = FlextContext.Correlation.get_correlation_id()
            assert current_correlation == correlation_id

        elif (
            test_case.operation == RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution(
                "cache", resolved=False, error_message="Connection refused"
            )
            assert FlextContext.Correlation.get_correlation_id() == correlation_id

        elif (
            test_case.operation
            == RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_domain_event(
                "UserCreated",
                aggregate_id="user-123",
                event_data={"email": "test@example.com"},
            )
            assert FlextContext.Correlation.get_correlation_id() == correlation_id

        elif (
            test_case.operation
            == RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_domain_event(
                "SystemInitialized",
                event_data={"timestamp": "2025-01-01T00:00:00Z"},
            )
            assert FlextContext.Correlation.get_correlation_id() == correlation_id

        elif (
            test_case.operation
            == RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="test-service",
                service_version="1.0.0",
                enable_context_correlation=True,
            )
            assert FlextContext.Service.get_service_name() == "test-service"
            assert FlextContext.Service.get_service_version() == "1.0.0"
            correlation_id_result: str | None = (
                FlextContext.Correlation.get_correlation_id()
            )
            assert correlation_id_result is not None

        elif (
            test_case.operation
            == RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="minimal-service",
                enable_context_correlation=True,
            )
            assert FlextContext.Service.get_service_name() == "minimal-service"
            assert FlextContext.Correlation.get_correlation_id() is not None

        elif (
            test_case.operation
            == RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION
        ):
            FlextRuntime.configure_structlog()
            structlog.contextvars.unbind_contextvars("correlation_id")
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="no-correlation-service",
                service_version="2.0.0",
                enable_context_correlation=False,
            )
            assert FlextContext.Service.get_service_name() == "no-correlation-service"
            assert FlextContext.Correlation.get_correlation_id() is None


__all__ = ["TestFlextRuntime"]
