"""Refactored comprehensive tests for u - Layer 0.5 Runtime Utilities.

Module: flext_core.runtime
Scope: u - type guards, serialization, external library access, type introspection

Tests all functionality of u including:
- Type guards (phone, dict-like, list-like, JSON, identifier)
- Serialization utilities (safe_get_attribute)
- External library access (structlog, dependency_injector)
- Type introspection (extract_generic_args, is_sequence_type)
- Structlog configuration
- Integration scenarios

Uses Python 3.13 patterns, u, c,
and aggressive parametrization for DRY testing.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from types import GenericAlias, ModuleType
from typing import cast

import pytest
import structlog
from dependency_injector import containers, providers
from hypothesis import given, strategies as st

from flext_core import FlextContainer, FlextContext, u
from flext_tests import tm
from tests import c, m, p, s, t, x


class TestFlextRuntime:
    type RuntimeScenarioValue = (
        t.NormalizedValue
        | type[t.NormalizedValue]
        | tuple[type[t.NormalizedValue], ...]
        | ModuleType
        | GenericAlias
    )
    type RuntimeScenarioInput = (
        RuntimeScenarioValue
        | Mapping[str, Callable[[], t.NormalizedValue]]
        | Sequence[ModuleType]
    )

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

    @dataclass(frozen=True)
    class RuntimeTestCase:
        """Runtime test case definition with parametrization data."""

        name: str
        operation: TestFlextRuntime.RuntimeOperationType
        test_input: (
            t.NormalizedValue
            | type[t.NormalizedValue]
            | tuple[type[t.NormalizedValue], ...]
            | ModuleType
            | GenericAlias
            | Mapping[str, Callable[[], t.NormalizedValue]]
            | Sequence[ModuleType]
            | None
        ) = None
        expected_result: (
            t.NormalizedValue
            | type[t.NormalizedValue]
            | tuple[type[t.NormalizedValue], ...]
            | ModuleType
            | GenericAlias
            | Mapping[str, Callable[[], t.NormalizedValue]]
            | Sequence[ModuleType]
            | None
        ) = None
        should_reset_config: bool = False

    class RuntimeScenarios:
        """Centralized runtime test scenarios using runtime-built cases."""

        RuntimeOperationType: type[TestFlextRuntime.RuntimeOperationType]
        RuntimeTestCase: type[TestFlextRuntime.RuntimeTestCase]

        @classmethod
        def _runtime_operation_type(cls) -> type[TestFlextRuntime.RuntimeOperationType]:
            return cls.RuntimeOperationType

        @classmethod
        def _runtime_test_case(cls) -> type[TestFlextRuntime.RuntimeTestCase]:
            return cls.RuntimeTestCase

        @classmethod
        def dict_like_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="dict_like_empty",
                    operation=runtime_op_type.DICT_LIKE_VALID,
                    test_input={},
                    expected_result=True,
                ),
                runtime_test_case(
                    name="dict_like_single_key",
                    operation=runtime_op_type.DICT_LIKE_VALID,
                    test_input={"key": "value"},
                    expected_result=True,
                ),
                runtime_test_case(
                    name="dict_like_nested",
                    operation=runtime_op_type.DICT_LIKE_VALID,
                    test_input={"nested": {"dict": True}},
                    expected_result=True,
                ),
                runtime_test_case(
                    name="dict_like_invalid_list",
                    operation=runtime_op_type.DICT_LIKE_INVALID,
                    test_input=[],
                    expected_result=False,
                ),
                runtime_test_case(
                    name="dict_like_invalid_string",
                    operation=runtime_op_type.DICT_LIKE_INVALID,
                    test_input="string",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="dict_like_invalid_int",
                    operation=runtime_op_type.DICT_LIKE_INVALID,
                    test_input=123,
                    expected_result=False,
                ),
                runtime_test_case(
                    name="dict_like_invalid_none",
                    operation=runtime_op_type.DICT_LIKE_INVALID,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @classmethod
        def list_like_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="list_like_empty",
                    operation=runtime_op_type.LIST_LIKE_VALID,
                    test_input=[],
                    expected_result=True,
                ),
                runtime_test_case(
                    name="list_like_integers",
                    operation=runtime_op_type.LIST_LIKE_VALID,
                    test_input=[1, 2, 3],
                    expected_result=True,
                ),
                runtime_test_case(
                    name="list_like_strings",
                    operation=runtime_op_type.LIST_LIKE_VALID,
                    test_input=["a", "b", "c"],
                    expected_result=True,
                ),
                runtime_test_case(
                    name="list_like_invalid_dict",
                    operation=runtime_op_type.LIST_LIKE_INVALID,
                    test_input={},
                    expected_result=False,
                ),
                runtime_test_case(
                    name="list_like_invalid_string",
                    operation=runtime_op_type.LIST_LIKE_INVALID,
                    test_input="string",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="list_like_invalid_int",
                    operation=runtime_op_type.LIST_LIKE_INVALID,
                    test_input=123,
                    expected_result=False,
                ),
                runtime_test_case(
                    name="list_like_invalid_none",
                    operation=runtime_op_type.LIST_LIKE_INVALID,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @classmethod
        def json_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="json_valid_object",
                    operation=runtime_op_type.JSON_VALID,
                    test_input='{"key": "value"}',
                    expected_result=True,
                ),
                runtime_test_case(
                    name="json_valid_empty_array",
                    operation=runtime_op_type.JSON_VALID,
                    test_input="[]",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="json_valid_array",
                    operation=runtime_op_type.JSON_VALID,
                    test_input="[1, 2, 3]",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="json_valid_string",
                    operation=runtime_op_type.JSON_VALID,
                    test_input='"string"',
                    expected_result=True,
                ),
                runtime_test_case(
                    name="json_valid_null",
                    operation=runtime_op_type.JSON_VALID,
                    test_input="null",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="json_invalid_plain_text",
                    operation=runtime_op_type.JSON_INVALID,
                    test_input="not json",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="json_invalid_malformed",
                    operation=runtime_op_type.JSON_INVALID,
                    test_input="{invalid}",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="json_invalid_empty",
                    operation=runtime_op_type.JSON_INVALID,
                    test_input="",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="json_non_string_dict",
                    operation=runtime_op_type.JSON_NON_STRING,
                    test_input={"key": "value"},
                    expected_result=False,
                ),
                runtime_test_case(
                    name="json_non_string_list",
                    operation=runtime_op_type.JSON_NON_STRING,
                    test_input=[1, 2, 3],
                    expected_result=False,
                ),
                runtime_test_case(
                    name="json_non_string_none",
                    operation=runtime_op_type.JSON_NON_STRING,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @classmethod
        def identifier_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="identifier_valid_lowercase",
                    operation=runtime_op_type.IDENTIFIER_VALID,
                    test_input="variable",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="identifier_valid_private",
                    operation=runtime_op_type.IDENTIFIER_VALID,
                    test_input="_private",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="identifier_valid_class_name",
                    operation=runtime_op_type.IDENTIFIER_VALID,
                    test_input="ClassName",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="identifier_valid_snake_case",
                    operation=runtime_op_type.IDENTIFIER_VALID,
                    test_input="function_name",
                    expected_result=True,
                ),
                runtime_test_case(
                    name="identifier_invalid_starts_with_digit",
                    operation=runtime_op_type.IDENTIFIER_INVALID,
                    test_input="123invalid",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="identifier_invalid_hyphen",
                    operation=runtime_op_type.IDENTIFIER_INVALID,
                    test_input="invalid-name",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="identifier_invalid_space",
                    operation=runtime_op_type.IDENTIFIER_INVALID,
                    test_input="invalid name",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="identifier_invalid_empty",
                    operation=runtime_op_type.IDENTIFIER_INVALID,
                    test_input="",
                    expected_result=False,
                ),
                runtime_test_case(
                    name="identifier_non_string_int",
                    operation=runtime_op_type.IDENTIFIER_NON_STRING,
                    test_input=123,
                    expected_result=False,
                ),
                runtime_test_case(
                    name="identifier_non_string_none",
                    operation=runtime_op_type.IDENTIFIER_NON_STRING,
                    test_input=None,
                    expected_result=False,
                ),
            ]

        @classmethod
        def generic_args_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="extract_generic_list",
                    operation=runtime_op_type.EXTRACT_GENERIC_GENERIC_TYPE,
                    test_input=list[str],
                    expected_result=(str,),
                ),
                runtime_test_case(
                    name="extract_generic_dict",
                    operation=runtime_op_type.EXTRACT_GENERIC_GENERIC_TYPE,
                    test_input=dict[str, int],
                    expected_result=(str, int),
                ),
                runtime_test_case(
                    name="extract_generic_non_generic_str",
                    operation=runtime_op_type.EXTRACT_GENERIC_NON_GENERIC,
                    test_input=str,
                    expected_result=(),
                ),
                runtime_test_case(
                    name="extract_generic_non_generic_int",
                    operation=runtime_op_type.EXTRACT_GENERIC_NON_GENERIC,
                    test_input=int,
                    expected_result=(),
                ),
                runtime_test_case(
                    name="extract_generic_exception_none",
                    operation=runtime_op_type.EXTRACT_GENERIC_EXCEPTION,
                    test_input=None,
                    expected_result=(),
                ),
                runtime_test_case(
                    name="extract_generic_exception_string",
                    operation=runtime_op_type.EXTRACT_GENERIC_EXCEPTION,
                    test_input="not a type",
                    expected_result=(),
                ),
            ]

        @classmethod
        def sequence_type_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="sequence_type_list_of_str",
                    operation=runtime_op_type.SEQUENCE_TYPE_VALID,
                    test_input=list[str],
                    expected_result=True,
                ),
                runtime_test_case(
                    name="sequence_type_tuple",
                    operation=runtime_op_type.SEQUENCE_TYPE_VALID,
                    test_input=tuple[int, ...],
                    expected_result=True,
                ),
                runtime_test_case(
                    name="sequence_type_str_is_sequence",
                    operation=runtime_op_type.SEQUENCE_TYPE_VALID,
                    test_input=str,
                    expected_result=True,
                ),
                runtime_test_case(
                    name="sequence_type_invalid_dict",
                    operation=runtime_op_type.SEQUENCE_TYPE_INVALID,
                    test_input=dict[str, int],
                    expected_result=False,
                ),
                runtime_test_case(
                    name="sequence_type_invalid_int",
                    operation=runtime_op_type.SEQUENCE_TYPE_INVALID,
                    test_input=int,
                    expected_result=False,
                ),
                runtime_test_case(
                    name="sequence_type_exception_none",
                    operation=runtime_op_type.SEQUENCE_TYPE_EXCEPTION,
                    test_input=None,
                    expected_result=False,
                ),
                runtime_test_case(
                    name="sequence_type_exception_string",
                    operation=runtime_op_type.SEQUENCE_TYPE_EXCEPTION,
                    test_input="not a type",
                    expected_result=False,
                ),
            ]

        @classmethod
        def serialization_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="safe_get_attribute_exists",
                    operation=runtime_op_type.SAFE_GET_ATTRIBUTE_EXISTS,
                    test_input=None,
                    expected_result="value",
                ),
                runtime_test_case(
                    name="safe_get_attribute_missing_with_default",
                    operation=runtime_op_type.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT,
                    test_input=None,
                    expected_result="default",
                ),
                runtime_test_case(
                    name="safe_get_attribute_missing_no_default",
                    operation=runtime_op_type.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT,
                    test_input=None,
                    expected_result=None,
                ),
            ]

        @classmethod
        def library_access_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="structlog_module",
                    operation=runtime_op_type.STRUCTLOG_MODULE,
                    test_input=None,
                    expected_result=structlog,
                ),
                runtime_test_case(
                    name="dependency_providers",
                    operation=runtime_op_type.DEPENDENCY_PROVIDERS_MODULE,
                    test_input=None,
                    expected_result=providers,
                ),
                runtime_test_case(
                    name="dependency_containers",
                    operation=runtime_op_type.DEPENDENCY_CONTAINERS_MODULE,
                    test_input=None,
                    expected_result=containers,
                ),
                runtime_test_case(
                    name="dependency_wiring_configuration",
                    operation=runtime_op_type.DEPENDENCY_WIRING_CONFIGURATION,
                ),
                runtime_test_case(
                    name="dependency_wiring_factories",
                    operation=runtime_op_type.DEPENDENCY_WIRING_FACTORIES,
                ),
                runtime_test_case(
                    name="dependency_wiring_automation",
                    operation=runtime_op_type.DEPENDENCY_WIRING_AUTOMATION,
                ),
                runtime_test_case(
                    name="service_runtime_automation",
                    operation=runtime_op_type.SERVICE_RUNTIME_AUTOMATION,
                ),
                runtime_test_case(
                    name="mixins_runtime_automation",
                    operation=runtime_op_type.MIXINS_RUNTIME_AUTOMATION,
                ),
            ]

        @classmethod
        def structlog_config_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="configure_defaults",
                    operation=runtime_op_type.CONFIGURE_STRUCTLOG_DEFAULTS,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                runtime_test_case(
                    name="configure_custom_level",
                    operation=runtime_op_type.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                runtime_test_case(
                    name="configure_json_renderer",
                    operation=runtime_op_type.CONFIGURE_STRUCTLOG_JSON_RENDERER,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                runtime_test_case(
                    name="configure_additional_processors",
                    operation=runtime_op_type.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
                runtime_test_case(
                    name="configure_idempotent",
                    operation=runtime_op_type.CONFIGURE_STRUCTLOG_IDEMPOTENT,
                    test_input=None,
                    expected_result=None,
                    should_reset_config=True,
                ),
            ]

        @classmethod
        def integration_scenarios(
            cls,
        ) -> Sequence[TestFlextRuntime.RuntimeTestCase]:
            runtime_test_case = cls._runtime_test_case()
            runtime_op_type = cls._runtime_operation_type()
            return [
                runtime_test_case(
                    name="constants_patterns",
                    operation=runtime_op_type.INTEGRATION_CONSTANTS_PATTERNS,
                ),
                runtime_test_case(
                    name="layer_hierarchy",
                    operation=runtime_op_type.INTEGRATION_LAYER_HIERARCHY,
                ),
                runtime_test_case(
                    name="track_service_resolution_success",
                    operation=runtime_op_type.TRACK_SERVICE_RESOLUTION_SUCCESS,
                ),
                runtime_test_case(
                    name="track_service_resolution_failure",
                    operation=runtime_op_type.TRACK_SERVICE_RESOLUTION_FAILURE,
                ),
                runtime_test_case(
                    name="track_domain_event_with_aggregate",
                    operation=runtime_op_type.TRACK_DOMAIN_EVENT_WITH_AGGREGATE,
                ),
                runtime_test_case(
                    name="track_domain_event_without_aggregate",
                    operation=runtime_op_type.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE,
                ),
                runtime_test_case(
                    name="setup_service_infrastructure_full",
                    operation=runtime_op_type.SETUP_SERVICE_INFRASTRUCTURE_FULL,
                ),
                runtime_test_case(
                    name="setup_service_infrastructure_minimal",
                    operation=runtime_op_type.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL,
                ),
                runtime_test_case(
                    name="setup_service_without_correlation",
                    operation=runtime_op_type.SETUP_SERVICE_WITHOUT_CORRELATION,
                ),
            ]

        @classmethod
        def reset_structlog_config(cls) -> None:
            """Reset structlog configuration for testing."""
            u._structlog_configured = False

    RuntimeScenarios.RuntimeOperationType = RuntimeOperationType
    RuntimeScenarios.RuntimeTestCase = RuntimeTestCase

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.dict_like_scenarios(),
        ids=lambda c: c.name,
    )
    def test_dict_like_validation(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test dict-like t.NormalizedValue validation.

        Business Rule: is_dict_like accepts t.NormalizedValue compatible objects.
        test_case.test_input may be None or various types, so we cast to t.NormalizedValue
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = u.is_dict_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result, eq=test_case.expected_result)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.list_like_scenarios(),
        ids=lambda c: c.name,
    )
    def test_list_like_validation(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test list-like t.NormalizedValue validation.

        Business Rule: is_list_like accepts t.NormalizedValue compatible objects.
        test_case.test_input may be None or various types, so we cast to t.NormalizedValue
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = u.is_list_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result, eq=test_case.expected_result)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.identifier_scenarios(),
        ids=lambda c: c.name,
    )
    def test_identifier_validation(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test Python identifier validation.

        Business Rule: None is a valid test input - validates that is_valid_identifier
        correctly returns False for None values.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = u.is_valid_identifier(
            cast("t.RuntimeData", test_case.test_input),
        )
        tm.that(result, eq=test_case.expected_result)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.serialization_scenarios(),
        ids=lambda c: c.name,
    )
    def test_safe_get_attribute(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test safe attribute retrieval."""
        if test_case.operation == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS:

            class TestObj:
                attr = "value"

            result = u.safe_get_attribute(TestObj, "attr")
            tm.that(result, eq="value")
        elif (
            test_case.operation
            == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT
        ):

            class TestObjDefault:
                pass

            result = u.safe_get_attribute(
                TestObjDefault,
                "missing",
                "default",
            )
            tm.that(result, eq="default")
        elif (
            test_case.operation
            == self.RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT
        ):

            class TestObjNoDefault:
                pass

            result = u.safe_get_attribute(TestObjNoDefault, "missing")
            tm.that(result, none=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.generic_args_scenarios(),
        ids=lambda c: c.name,
    )
    def test_extract_generic_args(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test extraction of generic type arguments.

        Business Rule: extract_generic_args accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        args = u.extract_generic_args(test_input_typed)
        tm.that(args, eq=test_case.expected_result)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.sequence_type_scenarios(),
        ids=lambda c: c.name,
    )
    def test_sequence_type_detection(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test sequence type detection.

        Business Rule: is_sequence_type accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        result = u.is_sequence_type(test_input_typed)
        tm.that(result, eq=test_case.expected_result)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.library_access_scenarios(),
        ids=lambda c: c.name,
    )
    def test_external_library_access(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test external library access."""
        if test_case.operation == self.RuntimeOperationType.STRUCTLOG_MODULE:
            module = u.structlog()
            tm.that(module is structlog, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["get_logger", "configure"]),
                eq=True,
            )
        elif (
            test_case.operation == self.RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE
        ):
            module = u.dependency_providers()
            tm.that(module is providers, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["Singleton", "Factory"]),
                eq=True,
            )
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE
        ):
            module = u.dependency_containers()
            tm.that(module is containers, eq=True)
            tm.that(hasattr(module, "DeclarativeContainer"), eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION
        ):
            di_container = u.DependencyIntegration.create_container()
            config_provider = u.DependencyIntegration.bind_configuration(
                di_container,
                t.ConfigMap(root={"database": {"dsn": "sqlite://"}}),
            )
            tm.that(repr(config_provider), ne="")
            config = di_container.config
            database_config = getattr(config, "database", None)
            if database_config is None:
                pytest.fail("database config should be available")
            dsn_value = database_config.dsn()
            tm.that(dsn_value, eq="sqlite://")
            module = ModuleType("di_config_module")

            @u.DependencyIntegration.inject
            def read_config(
                dsn: str = u.DependencyIntegration.Provide["config.database.dsn"],
            ) -> str:
                return dsn

            setattr(module, "read_config", read_config)
            di_container.wire(modules=[module])
            try:
                read_func = cast("Callable[[], str]", getattr(module, "read_config"))
                tm.that(callable(read_func), eq=True)
                result = read_func()
                tm.that(result, eq="sqlite://")
            finally:
                di_container.unwire()
        elif (
            test_case.operation == self.RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES
        ):
            di_container = u.DependencyIntegration.create_container()
            factory_provider = u.DependencyIntegration.register_factory(
                di_container,
                "token_factory",
                lambda: {"token": "abc123"},
            )
            object_provider = u.DependencyIntegration.register_object(
                di_container,
                "static_value",
                42,
            )
            assert isinstance(factory_provider, providers.Singleton)
            tm.that(factory_provider(), eq={"token": "abc123"})
            tm.that(object_provider(), eq=42)
            module = ModuleType("di_factory_module")

            @u.DependencyIntegration.inject
            def consume(
                token: t.StrMapping = u.DependencyIntegration.Provide["token_factory"],
                static: int = u.DependencyIntegration.Provide["static_value"],
            ) -> tuple[t.StrMapping, int]:
                return (token, static)

            setattr(module, "consume", consume)
            di_container.wire(modules=[module])
            try:
                consume_factory_func = cast(
                    "Callable[[], tuple[t.StrMapping, int]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_factory_func), eq=True)
                tokens, value = consume_factory_func()
                tm.that(tokens, eq={"token": "abc123"})
                tm.that(value, eq=42)
            finally:
                di_container.unwire()
        elif (
            test_case.operation
            == self.RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION
        ):
            counter = {"calls": 0}

            def token_factory() -> t.IntMapping:
                counter["calls"] += 1
                return {"token": counter["calls"]}

            module = ModuleType("di_automation_module")

            @u.DependencyIntegration.inject
            def consume_automation(
                static_value: int = u.DependencyIntegration.Provide["static_value"],
                token: t.IntMapping = u.DependencyIntegration.Provide["token_factory"],
                config_flag: bool = u.DependencyIntegration.Provide[
                    "config.flags.enabled"
                ],
                resource: Mapping[
                    str,
                    bool,
                ] = u.DependencyIntegration.Provide["api_client"],
            ) -> tuple[int, t.IntMapping, bool, t.BoolMapping]:
                return (static_value, token, config_flag, resource)

            setattr(module, "consume", consume_automation)
            di_container = u.DependencyIntegration.create_container(
                container_options=m.DependencyContainerCreationOptions(
                    config=t.ConfigMap(root={"flags": {"enabled": True}}),
                    services={"static_value": 7},
                    factories={"token_factory": token_factory},
                    resources={"api_client": lambda: {"connected": True}},
                    wire_modules=[module],
                    factory_cache=False,
                ),
            )
            try:
                consume_automation_func = cast(
                    "Callable[[], tuple[int, t.IntMapping, bool, t.BoolMapping]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_automation_func), eq=True)
                first_static, first_token, config_enabled, resource_value = (
                    consume_automation_func()
                )
                second_static, second_token, _, _ = consume_automation_func()
                tm.that(first_static, eq=second_static)
                tm.that(config_enabled is True, eq=True)
                tm.that(resource_value, eq={"connected": True})
                tm.that(first_token["token"], eq=1)
                tm.that(second_token["token"], eq=2)
            finally:
                di_container.unwire()
        elif (
            test_case.operation == self.RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION
        ):
            counter = {"calls": 0}

            def token_factory() -> t.IntMapping:
                counter["calls"] += 1
                return {"count": counter["calls"]}

            module = ModuleType("service_runtime_module")

            @u.DependencyIntegration.inject
            def consume_service(
                flag: bool = u.DependencyIntegration.Provide["feature_flag"],
                token: t.IntMapping = u.DependencyIntegration.Provide["token_factory"],
                resource: Mapping[
                    str,
                    bool,
                ] = u.DependencyIntegration.Provide["api_client"],
            ) -> tuple[bool, t.IntMapping, t.BoolMapping]:
                return (flag, token, resource)

            setattr(module, "consume", consume_service)
            runtime_raw = s._create_runtime(
                runtime_options=m.RuntimeBootstrapOptions(
                    config_overrides={"app_name": "runtime-service"},
                    services={"feature_flag": True},
                    factories={"token_factory": token_factory},
                    resources={"api_client": lambda: {"connected": True}},
                    wire_modules=[module],
                ),
            )
            runtime = runtime_raw
            try:
                consume_service_func = cast(
                    "Callable[[], tuple[bool, t.IntMapping, t.BoolMapping]]",
                    getattr(module, "consume"),
                )
                tm.that(callable(consume_service_func), eq=True)
                feature_flag, first_token, resource = consume_service_func()
                _, second_token, _ = consume_service_func()
                tm.that(runtime.config.app_name, eq="runtime-service")
                tm.that(feature_flag is True, eq=True)
                tm.that(resource, eq={"connected": True})
                tm.that(first_token["count"], eq=1)
                tm.that(second_token["count"], eq=2)
            finally:
                container = cast("FlextContainer", runtime.container)
                container._di_bridge.unwire()
        elif test_case.operation == self.RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION:

            class RuntimeAwareComponent(x):
                @classmethod
                def _runtime_bootstrap_options(cls) -> m.RuntimeBootstrapOptions:

                    def counter_factory() -> t.IntMapping:
                        return {"count": 1}

                    return m.RuntimeBootstrapOptions(
                        config_overrides={"app_name": "runtime-aware"},
                        services={"preseed": {"enabled": True}},
                        factories={"counter": counter_factory},
                    )

            component = RuntimeAwareComponent(
                config_type=None,
                config_overrides=None,
                initial_context=None,
            )
            runtime_first = component._get_runtime()
            runtime_second = component._get_runtime()
            tm.that(runtime_first is runtime_second, eq=True)
            tm.that(component.config.app_name, eq="runtime-aware")
            tm.that(component.context is runtime_first.context, eq=True)
            service_result = component.container.get("preseed", type_cls=t.ConfigMap)
            tm.that(service_result.is_success, eq=True)
            tm.that(service_result.value, eq=t.ConfigMap(root={"enabled": True}))
            factory_result = component.container.get("counter")
            assert factory_result.is_success
            assert factory_result.value == {"count": 1}

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.structlog_config_scenarios(),
        ids=lambda c: c.name,
    )
    def test_structlog_configuration(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test structlog configuration."""
        if test_case.should_reset_config:
            self.RuntimeScenarios.reset_structlog_config()
        if (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS
        ):
            u.configure_structlog()
            tm.that(u._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL
        ):
            u.configure_structlog(log_level=logging.DEBUG)
            tm.that(u._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER
        ):
            u.configure_structlog(console_renderer=False)
            tm.that(u._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS
        ):

            def custom_processor(
                _logger: p.Logger,
                _method_name: str,
                event_dict: t.MutableContainerMapping,
            ) -> t.MutableContainerMapping:
                event_dict["custom"] = True
                return event_dict

            u.configure_structlog(additional_processors=[custom_processor])
            tm.that(u._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT
        ):
            u.configure_structlog()
            u.configure_structlog()
            tm.that(u._structlog_configured is True, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.integration_scenarios(),
        ids=lambda c: c.name,
    )
    def test_runtime_integration(
        self,
        test_case: TestFlextRuntime.RuntimeTestCase,
    ) -> None:
        """Test u integration scenarios."""
        if (
            test_case.operation
            == self.RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS
        ):
            tm.that(hasattr(c, "PATTERN_PHONE_NUMBER"), eq=True)
        elif (
            test_case.operation == self.RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY
        ):
            tm.that(hasattr(c, "PATTERN_EMAIL"), eq=True)
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS
        ):
            u.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            u.Integration.track_service_resolution("database", resolved=True)
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=correlation_id)
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE
        ):
            u.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            u.Integration.track_service_resolution(
                "cache",
                resolved=False,
                error_message="Connection refused",
            )
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=correlation_id)
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE
        ):
            u.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            u.Integration.track_domain_event(
                "UserCreated",
                aggregate_id="user-123",
                event_data=t.ConfigMap(root={"email": "test@example.com"}),
            )
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=correlation_id)
        elif (
            test_case.operation
            == self.RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE
        ):
            u.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            u.Integration.track_domain_event(
                "SystemInitialized",
                event_data=t.ConfigMap(root={"timestamp": "2025-01-01T00:00:00Z"}),
            )
            tm.that(FlextContext.Correlation.get_correlation_id(), eq=correlation_id)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL
        ):
            u.configure_structlog()
            u.Integration.setup_service_infrastructure(
                service_name="test-service",
                service_version="1.0.0",
                enable_context_correlation=True,
            )
            tm.that(FlextContext.Variables.ServiceName.get(), eq="test-service")
            tm.that(FlextContext.Variables.ServiceVersion.get(), eq="1.0.0")
            tm.that(FlextContext.Correlation.get_correlation_id(), none=False)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL
        ):
            u.configure_structlog()
            u.Integration.setup_service_infrastructure(
                service_name="minimal-service",
                enable_context_correlation=True,
            )
            tm.that(FlextContext.Variables.ServiceName.get(), eq="minimal-service")
            tm.that(FlextContext.Correlation.get_correlation_id(), none=False)
        elif (
            test_case.operation
            == self.RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION
        ):
            u.configure_structlog()
            structlog.contextvars.unbind_contextvars("correlation_id")
            u.Integration.setup_service_infrastructure(
                service_name="no-correlation-service",
                service_version="2.0.0",
                enable_context_correlation=False,
            )
            tm.that(
                FlextContext.Variables.ServiceName.get(),
                eq="no-correlation-service",
            )
            tm.that(FlextContext.Correlation.get_correlation_id(), none=True)

    @given(st.text())
    def test_hypothesis_identifier_guard_returns_bool(self, value: str) -> None:
        """Property: is_valid_identifier always returns bool."""
        result = u.is_valid_identifier(value)
        tm.that(result, is_=bool)

    @given(
        st.one_of(
            st.integers(),
            st.text(),
            st.floats(allow_nan=False, allow_infinity=False),
            st.booleans(),
        ),
    )
    def test_hypothesis_type_guards_return_bool(
        self,
        value: float | str | bool,
    ) -> None:
        """Property: type guards always return bool for any input."""
        tm.that(u.is_dict_like(value), is_=bool)
        tm.that(u.is_list_like(value), is_=bool)

    __all__ = ["TestFlextRuntime"]


__all__ = ["TestFlextRuntime"]
