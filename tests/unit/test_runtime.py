"""Refactored comprehensive tests for FlextRuntime - Layer 0.5 Runtime Utilities.

Module: flext_core.runtime
Scope: FlextRuntime - type guards, serialization, external library access, type introspection

Tests all functionality of FlextRuntime including:
- Type guards (phone, dict-like, list-like, JSON, identifier)
- Serialization utilities (safe_get_attribute)
- External library access (structlog, dependency_injector)
- Type introspection (extract_generic_args, is_sequence_type)
- Structlog configuration
- Integration scenarios

Uses Python 3.13 patterns, FlextTestsUtilities, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from enum import StrEnum, unique
from types import ModuleType
from typing import Annotated, cast

import pytest
import structlog
from dependency_injector import containers, providers
from flext_tests import t, tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import FlextContainer, FlextContext, FlextRuntime, c, m, p, s, x


class TestFlextRuntime:
    @unique
    class RuntimeOperationType(StrEnum):
        """Runtime operation types for parametrized testing."""

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
        SAFE_GET_ATTRIBUTE_EXISTS = "safe_get_attribute_exists"
        SAFE_GET_ATTRIBUTE_MISSING_DEFAULT = "safe_get_attribute_missing_default"
        SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT = "safe_get_attribute_missing_no_default"
        EXTRACT_GENERIC_GENERIC_TYPE = "extract_generic_generic_type"
        EXTRACT_GENERIC_NON_GENERIC = "extract_generic_non_generic"
        EXTRACT_GENERIC_EXCEPTION = "extract_generic_exception"
        SEQUENCE_TYPE_VALID = "sequence_type_valid"
        SEQUENCE_TYPE_INVALID = "sequence_type_invalid"
        SEQUENCE_TYPE_EXCEPTION = "sequence_type_exception"
        STRUCTLOG_MODULE = "structlog_module"
        DEPENDENCY_PROVIDERS_MODULE = "dependency_providers_module"
        DEPENDENCY_CONTAINERS_MODULE = "dependency_containers_module"
        DEPENDENCY_WIRING_CONFIGURATION = "dependency_wiring_configuration"
        DEPENDENCY_WIRING_FACTORIES = "dependency_wiring_factories"
        DEPENDENCY_WIRING_AUTOMATION = "dependency_wiring_automation"
        SERVICE_RUNTIME_AUTOMATION = "service_runtime_automation"
        MIXINS_RUNTIME_AUTOMATION = "mixins_runtime_automation"
        CONFIGURE_STRUCTLOG_DEFAULTS = "configure_structlog_defaults"
        CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL = "configure_structlog_custom_log_level"
        CONFIGURE_STRUCTLOG_JSON_RENDERER = "configure_structlog_json_renderer"
        CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS = (
            "configure_structlog_additional_processors"
        )
        CONFIGURE_STRUCTLOG_IDEMPOTENT = "configure_structlog_idempotent"
        INTEGRATION_CONSTANTS_PATTERNS = "integration_constants_patterns"
        INTEGRATION_LAYER_HIERARCHY = "integration_layer_hierarchy"
        TRACK_SERVICE_RESOLUTION_SUCCESS = "track_service_resolution_success"
        TRACK_SERVICE_RESOLUTION_FAILURE = "track_service_resolution_failure"
        TRACK_DOMAIN_EVENT_WITH_AGGREGATE = "track_domain_event_with_aggregate"
        TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE = "track_domain_event_without_aggregate"
        SETUP_SERVICE_INFRASTRUCTURE_FULL = "setup_service_infrastructure_full"
        SETUP_SERVICE_INFRASTRUCTURE_MINIMAL = "setup_service_infrastructure_minimal"
        SETUP_SERVICE_WITHOUT_CORRELATION = "setup_service_without_correlation"

    class RuntimeTestCase(BaseModel):
        """Runtime test case definition with parametrization data."""

        model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
        name: Annotated[str, Field(description="Runtime test case name")]
        operation: Annotated[StrEnum, Field(description="Runtime operation type")]
        test_input: Annotated[
            object, Field(default=None, description="Optional test input")
        ] = None
        expected_result: Annotated[
            object, Field(default=None, description="Expected operation result")
        ] = None
        should_reset_config: Annotated[
            bool,
            Field(
                default=False,
                description="Whether structlog config should be reset before test",
            ),
        ] = False

    globals()["RuntimeOperationType"] = RuntimeOperationType
    globals()["RuntimeTestCase"] = RuntimeTestCase

    class RuntimeScenarios:
        """Centralized runtime test scenarios using runtime-built cases."""

        @staticmethod
        def dict_like_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_empty",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_VALID,
                    test_input={},
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_single_key",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_VALID,
                    test_input={"key": "value"},
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_nested",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_VALID,
                    test_input={"nested": {"dict": True}},
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_invalid_list",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_INVALID,
                    test_input=[],
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_invalid_string",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_INVALID,
                    test_input="string",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_invalid_int",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_INVALID,
                    test_input=123,
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dict_like_invalid_none",
                    operation=TestFlextRuntime.RuntimeOperationType.DICT_LIKE_INVALID,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @staticmethod
        def list_like_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_empty",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_VALID,
                    test_input=[],
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_integers",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_VALID,
                    test_input=[1, 2, 3],
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_strings",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_VALID,
                    test_input=["a", "b", "c"],
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_invalid_dict",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_INVALID,
                    test_input={},
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_invalid_string",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_INVALID,
                    test_input="string",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_invalid_int",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_INVALID,
                    test_input=123,
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="list_like_invalid_none",
                    operation=TestFlextRuntime.RuntimeOperationType.LIST_LIKE_INVALID,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @staticmethod
        def json_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="json_valid_object",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_VALID,
                    test_input='{"key": "value"}',
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_valid_empty_array",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_VALID,
                    test_input="[]",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_valid_array",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_VALID,
                    test_input="[1, 2, 3]",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_valid_string",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_VALID,
                    test_input='"string"',
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_valid_null",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_VALID,
                    test_input="null",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_invalid_plain_text",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_INVALID,
                    test_input="not json",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_invalid_malformed",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_INVALID,
                    test_input="{invalid}",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_invalid_empty",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_INVALID,
                    test_input="",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_non_string_dict",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_NON_STRING,
                    test_input={"key": "value"},
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_non_string_list",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_NON_STRING,
                    test_input=[1, 2, 3],
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="json_non_string_none",
                    operation=TestFlextRuntime.RuntimeOperationType.JSON_NON_STRING,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @staticmethod
        def identifier_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_valid_lowercase",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_VALID,
                    test_input="variable",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_valid_private",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_VALID,
                    test_input="_private",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_valid_class_name",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_VALID,
                    test_input="ClassName",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_valid_snake_case",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_VALID,
                    test_input="function_name",
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_invalid_starts_with_digit",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_INVALID,
                    test_input="123invalid",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_invalid_hyphen",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_INVALID,
                    test_input="invalid-name",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_invalid_space",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_INVALID,
                    test_input="invalid name",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_invalid_empty",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_INVALID,
                    test_input="",
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_non_string_int",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_NON_STRING,
                    test_input=123,
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="identifier_non_string_none",
                    operation=TestFlextRuntime.RuntimeOperationType.IDENTIFIER_NON_STRING,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @staticmethod
        def generic_args_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_list",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
                    test_input=list[str],
                    expected_result=(str,),
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_dict",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
                    test_input=dict[str, int],
                    expected_result=(str, int),
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_non_generic_str",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
                    test_input=str,
                    expected_result=(),
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_non_generic_int",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
                    test_input=int,
                    expected_result=(),
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_exception_none",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
                    test_input=None,
                    expected_result=(),
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="extract_generic_exception_string",
                    operation=TestFlextRuntime.RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
                    test_input="not a type",
                    expected_result=(),
                ),
            ]

        @staticmethod
        def sequence_type_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_list_of_str",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_VALID,
                    test_input=list[str],
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_tuple",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_VALID,
                    test_input=tuple[int, ...],
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_str_is_sequence",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_VALID,
                    test_input=str,
                    expected_result=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_invalid_dict",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_INVALID,
                    test_input=dict[str, int],
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_invalid_int",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_INVALID,
                    test_input=int,
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_exception_none",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
                    test_input=None,
                    expected_result=False,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="sequence_type_exception_string",
                    operation=TestFlextRuntime.RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
                    test_input="not a type",
                    expected_result=False,
                ),
            ]

        @staticmethod
        def serialization_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="safe_get_attribute_exists",
                    operation=TestFlextRuntime.RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS,
                    test_input=None,
                    expected_result="value",
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="safe_get_attribute_missing_with_default",
                    operation=TestFlextRuntime.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT,
                    test_input=None,
                    expected_result="default",
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="safe_get_attribute_missing_no_default",
                    operation=TestFlextRuntime.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT,
                    test_input=None,
                    expected_result=None,
                ),
            ]

        @staticmethod
        def library_access_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="structlog_module",
                    operation=TestFlextRuntime.RuntimeOperationType.STRUCTLOG_MODULE,
                    test_input=None,
                    expected_result=structlog,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dependency_providers",
                    operation=TestFlextRuntime.RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE,
                    test_input=None,
                    expected_result=providers,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dependency_containers",
                    operation=TestFlextRuntime.RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE,
                    test_input=None,
                    expected_result=containers,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dependency_wiring_configuration",
                    operation=TestFlextRuntime.RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dependency_wiring_factories",
                    operation=TestFlextRuntime.RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="dependency_wiring_automation",
                    operation=TestFlextRuntime.RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="service_runtime_automation",
                    operation=TestFlextRuntime.RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="mixins_runtime_automation",
                    operation=TestFlextRuntime.RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION,
                ),
            ]

        @staticmethod
        def structlog_config_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="configure_defaults",
                    operation=TestFlextRuntime.RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="configure_custom_level",
                    operation=TestFlextRuntime.RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="configure_json_renderer",
                    operation=TestFlextRuntime.RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="configure_additional_processors",
                    operation=TestFlextRuntime.RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="configure_idempotent",
                    operation=TestFlextRuntime.RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
            ]

        @staticmethod
        def integration_scenarios() -> list[TestFlextRuntime.RuntimeTestCase]:
            return [
                TestFlextRuntime.RuntimeTestCase(
                    name="constants_patterns",
                    operation=TestFlextRuntime.RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="layer_hierarchy",
                    operation=TestFlextRuntime.RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="track_service_resolution_success",
                    operation=TestFlextRuntime.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="track_service_resolution_failure",
                    operation=TestFlextRuntime.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="track_domain_event_with_aggregate",
                    operation=TestFlextRuntime.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="track_domain_event_without_aggregate",
                    operation=TestFlextRuntime.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="setup_service_infrastructure_full",
                    operation=TestFlextRuntime.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="setup_service_infrastructure_minimal",
                    operation=TestFlextRuntime.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL,
                ),
                TestFlextRuntime.RuntimeTestCase(
                    name="setup_service_without_correlation",
                    operation=TestFlextRuntime.RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION,
                ),
            ]

        @staticmethod
        def reset_structlog_config() -> None:
            """Reset structlog configuration for testing."""
            FlextRuntime._structlog_configured = False

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.dict_like_scenarios(), ids=lambda c: c.name
    )
    def test_dict_like_validation(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test dict-like object validation.

        Business Rule: is_dict_like accepts object compatible objects.
        test_case.test_input may be None or various types, so we cast to object
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_dict_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.list_like_scenarios(), ids=lambda c: c.name
    )
    def test_list_like_validation(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test list-like object validation.

        Business Rule: is_list_like accepts object compatible objects.
        test_case.test_input may be None or various types, so we cast to object
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_list_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.json_scenarios(), ids=lambda c: c.name
    )
    def test_json_validation(self, test_case: TestFlextRuntime.RuntimeTestCase) -> None:
        """Test JSON string validation.

        Business Rule: None is a valid test input - validates that is_valid_json
        correctly returns False for None values.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_valid_json(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.identifier_scenarios(), ids=lambda c: c.name
    )
    def test_identifier_validation(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test Python identifier validation.

        Business Rule: None is a valid test input - validates that is_valid_identifier
        correctly returns False for None values.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_valid_identifier(
            cast("t.RuntimeData", test_case.test_input)
        )
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.serialization_scenarios(), ids=lambda c: c.name
    )
    def test_safe_get_attribute(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test safe attribute retrieval."""
        if test_case.operation == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS:

            class TestObj:
                attr = "value"

            result = FlextRuntime.safe_get_attribute(TestObj, "attr")
            tm.that(result == "value", eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT
        ):

            class TestObjDefault:
                pass

            result = FlextRuntime.safe_get_attribute(
                TestObjDefault, "missing", "default"
            )
            tm.that(result == "default", eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT
        ):

            class TestObjNoDefault:
                pass

            result = FlextRuntime.safe_get_attribute(TestObjNoDefault, "missing")
            tm.that(result is None, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.generic_args_scenarios(), ids=lambda c: c.name
    )
    def test_extract_generic_args(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test extraction of generic type arguments.

        Business Rule: extract_generic_args accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        args = FlextRuntime.extract_generic_args(test_input_typed)
        tm.that(args == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.sequence_type_scenarios(), ids=lambda c: c.name
    )
    def test_sequence_type_detection(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test sequence type detection.

        Business Rule: is_sequence_type accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        result = FlextRuntime.is_sequence_type(test_input_typed)
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.library_access_scenarios(), ids=lambda c: c.name
    )
    def test_external_library_access(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test external library access."""
        if test_case.operation == self.RuntimeOperationType.STRUCTLOG_MODULE:
            module = FlextRuntime.structlog()
            tm.that(module is structlog, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["get_logger", "configure"]),
                eq=True,
            )
        elif (
            test_case.operation == self.RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE
        ):
            module = FlextRuntime.dependency_providers()
            tm.that(module is providers, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["Singleton", "Factory"]), eq=True
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE
        ):
            module = FlextRuntime.dependency_containers()
            tm.that(module is containers, eq=True)
            tm.that(hasattr(module, "DeclarativeContainer"), eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION
        ):
            di_container = FlextRuntime.DependencyIntegration.create_container()
            config_provider = FlextRuntime.DependencyIntegration.bind_configuration(
                di_container, t.ConfigMap(root={"database": {"dsn": "sqlite://"}})
            )
            tm.that(repr(config_provider) != "", eq=True)
            config = di_container.config
            database_config = getattr(config, "database", None)
            if database_config is None:
                pytest.fail("database config should be available")
            dsn_value = database_config.dsn()
            tm.that(dsn_value == "sqlite://", eq=True)
            module = ModuleType("di_config_module")

            @FlextRuntime.DependencyIntegration.inject
            def read_config(
                dsn: str = FlextRuntime.DependencyIntegration.Provide[
                    "config.database.dsn"
                ],
            ) -> str:
                return dsn

            setattr(module, "read_config", read_config)
            di_container.wire(modules=[module])
            try:
                read_func = cast("Callable[[], str]", getattr(module, "read_config"))
                tm.that(callable(read_func), eq=True)
                result = read_func()
                tm.that(result == "sqlite://", eq=True)
            finally:
                di_container.unwire()
        elif (
            test_case.operation == self.RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES
        ):
            di_container = FlextRuntime.DependencyIntegration.create_container()
            factory_provider = FlextRuntime.DependencyIntegration.register_factory(
                di_container, "token_factory", lambda: {"token": "abc123"}
            )
            object_provider = FlextRuntime.DependencyIntegration.register_object(
                di_container, "static_value", 42
            )
            tm.that(isinstance(factory_provider, providers.Singleton), eq=True)
            tm.that(factory_provider() == {"token": "abc123"}, eq=True)
            tm.that(object_provider() == 42, eq=True)
            module = ModuleType("di_factory_module")

            @FlextRuntime.DependencyIntegration.inject
            def consume(
                token: dict[str, str] = FlextRuntime.DependencyIntegration.Provide[
                    "token_factory"
                ],
                static: int = FlextRuntime.DependencyIntegration.Provide[
                    "static_value"
                ],
            ) -> tuple[dict[str, str], int]:
                return (token, static)

            setattr(module, "consume", consume)
            di_container.wire(modules=[module])
            try:
                consume_factory_func = cast(
                    "Callable[[], tuple[dict[str, str], int]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_factory_func), eq=True)
                tokens, value = consume_factory_func()
                tm.that(tokens == {"token": "abc123"}, eq=True)
                tm.that(value == 42, eq=True)
            finally:
                di_container.unwire()
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION
        ):
            counter = {"calls": 0}

            def token_factory() -> dict[str, int]:
                counter["calls"] += 1
                return {"token": counter["calls"]}

            module = ModuleType("di_automation_module")

            @FlextRuntime.DependencyIntegration.inject
            def consume_automation(
                static_value: int = FlextRuntime.DependencyIntegration.Provide[
                    "static_value"
                ],
                token: dict[str, int] = FlextRuntime.DependencyIntegration.Provide[
                    "token_factory"
                ],
                config_flag: bool = FlextRuntime.DependencyIntegration.Provide[
                    "config.flags.enabled"
                ],
                resource: dict[str, bool] = FlextRuntime.DependencyIntegration.Provide[
                    "api_client"
                ],
            ) -> tuple[int, dict[str, int], bool, dict[str, bool]]:
                return (static_value, token, config_flag, resource)

            setattr(module, "consume", consume_automation)
            di_container = FlextRuntime.DependencyIntegration.create_container(
                config=t.ConfigMap(root={"flags": {"enabled": True}}),
                services={"static_value": 7},
                factories={"token_factory": token_factory},
                resources={"api_client": lambda: {"connected": True}},
                wire_modules=[module],
                factory_cache=False,
            )
            try:
                consume_automation_func = cast(
                    "Callable[[], tuple[int, dict[str, int], bool, dict[str, bool]]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_automation_func), eq=True)
                first_static, first_token, config_enabled, resource_value = (
                    consume_automation_func()
                )
                second_static, second_token, _, _ = consume_automation_func()
                tm.that(first_static == second_static == 7, eq=True)
                tm.that(config_enabled is True, eq=True)
                tm.that(resource_value == {"connected": True}, eq=True)
                tm.that(first_token["token"] == 1, eq=True)
                tm.that(second_token["token"] == 2, eq=True)
            finally:
                di_container.unwire()
        elif (
            test_case.operation == self.RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION
        ):
            counter = {"calls": 0}

            def token_factory() -> dict[str, int]:
                counter["calls"] += 1
                return {"count": counter["calls"]}

            module = ModuleType("service_runtime_module")

            @FlextRuntime.DependencyIntegration.inject
            def consume_service(
                flag: bool = FlextRuntime.DependencyIntegration.Provide["feature_flag"],
                token: dict[str, int] = FlextRuntime.DependencyIntegration.Provide[
                    "token_factory"
                ],
                resource: dict[str, bool] = FlextRuntime.DependencyIntegration.Provide[
                    "api_client"
                ],
            ) -> tuple[bool, dict[str, int], dict[str, bool]]:
                return (flag, token, resource)

            setattr(module, "consume", consume_service)
            runtime_raw = s._create_runtime(
                config_overrides={"app_name": "runtime-service"},
                services={"feature_flag": True},
                factories={"token_factory": token_factory},
                resources={"api_client": lambda: {"connected": True}},
                wire_modules=[module],
            )
            runtime = runtime_raw
            try:
                consume_service_func = cast(
                    "Callable[[], tuple[bool, dict[str, int], dict[str, bool]]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_service_func), eq=True)
                feature_flag, first_token, resource = consume_service_func()
                _, second_token, _ = consume_service_func()
                tm.that(runtime.config.app_name == "runtime-service", eq=True)
                tm.that(feature_flag is True, eq=True)
                tm.that(resource == {"connected": True}, eq=True)
                tm.that(first_token["count"] == 1, eq=True)
                tm.that(second_token["count"] == 2, eq=True)
            finally:
                container = cast("FlextContainer", runtime.container)
                container._di_bridge.unwire()
        elif test_case.operation == self.RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION:

            class RuntimeAwareComponent(x):
                @classmethod
                def _runtime_bootstrap_options(cls) -> m.RuntimeBootstrapOptions:

                    def counter_factory() -> dict[str, int]:
                        return {"count": 1}

                    return m.RuntimeBootstrapOptions(
                        config_overrides={"app_name": "runtime-aware"},
                        services={"preseed": {"enabled": True}},
                        factories={"counter": counter_factory},
                    )

            component = RuntimeAwareComponent(
                config_type=None, config_overrides=None, initial_context=None
            )
            runtime_first = component._get_runtime()
            runtime_second = component._get_runtime()
            tm.that(runtime_first is runtime_second, eq=True)
            tm.that(component.config.app_name == "runtime-aware", eq=True)
            tm.that(component.context is runtime_first.context, eq=True)
            service_result = component.container.get("preseed")
            tm.that(service_result.is_success, eq=True)
            tm.that(
                service_result.value == t.ConfigMap(root={"enabled": True}), eq=True
            )
            factory_result = component.container.get("counter")
            tm.that(factory_result.is_success, eq=True)
            tm.that(factory_result.value == {"count": 1}, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.structlog_config_scenarios(), ids=lambda c: c.name
    )
    def test_structlog_configuration(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test structlog configuration."""
        if test_case.should_reset_config:
            self.RuntimeScenarios.reset_structlog_config()
        if (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS
        ):
            FlextRuntime.configure_structlog()
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL
        ):
            FlextRuntime.configure_structlog(log_level=logging.DEBUG)
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER
        ):
            FlextRuntime.configure_structlog(console_renderer=False)
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS
        ):

            def custom_processor(
                _logger: p.Logger,
                _method_name: str,
                event_dict: dict[str, t.Tests.object],
            ) -> dict[str, t.Tests.object]:
                event_dict["custom"] = True
                return event_dict

            _ = custom_processor
            FlextRuntime.configure_structlog(additional_processors=["custom_processor"])
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.configure_structlog()
            tm.that(FlextRuntime._structlog_configured is True, eq=True)

    @pytest.mark.parametrize(
        "test_case", RuntimeScenarios.integration_scenarios(), ids=lambda c: c.name
    )
    def test_runtime_integration(
        self, test_case: TestFlextRuntime.RuntimeTestCase
    ) -> None:
        """Test FlextRuntime integration scenarios."""
        if (
            test_case.operation
            == self.RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS
        ):
            tm.that(hasattr(c.Platform, "PATTERN_PHONE_NUMBER"), eq=True)
            tm.that(FlextRuntime.is_valid_json('{"phone": "+5511987654321"}'), eq=True)
        elif (
            test_case.operation == self.RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY
        ):
            tm.that(hasattr(c.Platform, "PATTERN_EMAIL"), eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution("database", resolved=True)
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution(
                "cache", resolved=False, error_message="Connection refused"
            )
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_domain_event(
                "UserCreated",
                aggregate_id="user-123",
                event_data=t.ConfigMap(root={"email": "test@example.com"}),
            )
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_domain_event(
                "SystemInitialized",
                event_data=t.ConfigMap(root={"timestamp": "2025-01-01T00:00:00Z"}),
            )
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="test-service",
                service_version="1.0.0",
                enable_context_correlation=True,
            )
            tm.that(FlextContext.Variables.ServiceName.get() == "test-service", eq=True)
            tm.that(FlextContext.Variables.ServiceVersion.get() == "1.0.0", eq=True)
            tm.that(FlextContext.Correlation.get_correlation_id() is not None, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="minimal-service", enable_context_correlation=True
            )
            tm.that(
                FlextContext.Variables.ServiceName.get() == "minimal-service", eq=True
            )
            tm.that(FlextContext.Correlation.get_correlation_id() is not None, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION
        ):
            FlextRuntime.configure_structlog()
            structlog.contextvars.unbind_contextvars("correlation_id")
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="no-correlation-service",
                service_version="2.0.0",
                enable_context_correlation=False,
            )
            tm.that(
                FlextContext.Variables.ServiceName.get() == "no-correlation-service",
                eq=True,
            )
            tm.that(FlextContext.Correlation.get_correlation_id() is None, eq=True)

    __all__ = ["TestFlextRuntime"]


__all__ = ["TestFlextRuntime"]
