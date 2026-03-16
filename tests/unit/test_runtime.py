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
from enum import StrEnum
from types import ModuleType
from typing import Annotated, ClassVar, cast

import pytest
import structlog
from dependency_injector import containers, providers
from flext_tests import t, tm
from pydantic import BaseModel, ConfigDict, Field

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextRuntime,
    c,
    m,
    p,
    s,
    x,
)


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
    operation: Annotated[
        RuntimeOperationType, Field(description="Runtime operation type")
    ]
    test_input: Annotated[
        object,
        Field(
            default=None,
            description="Optional test input",
        ),
    ] = None
    expected_result: Annotated[
        object,
        Field(
            default=None,
            description="Expected operation result",
        ),
    ] = None
    should_reset_config: Annotated[
        bool,
        Field(
            default=False,
            description="Whether structlog config should be reset before test",
        ),
    ] = False


class RuntimeScenarios:
    """Centralized runtime test scenarios using c."""

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
            test_input=None,
            expected_result="value",
        ),
        RuntimeTestCase(
            name="safe_get_attribute_missing_with_default",
            operation=RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT,
            test_input=None,
            expected_result="default",
        ),
        RuntimeTestCase(
            name="safe_get_attribute_missing_no_default",
            operation=RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT,
            test_input=None,
            expected_result=None,
        ),
    ]
    LIBRARY_ACCESS_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="structlog_module",
            operation=RuntimeOperationType.STRUCTLOG_MODULE,
            test_input=None,
            expected_result=structlog,
        ),
        RuntimeTestCase(
            name="dependency_providers",
            operation=RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE,
            test_input=None,
            expected_result=providers,
        ),
        RuntimeTestCase(
            name="dependency_containers",
            operation=RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE,
            test_input=None,
            expected_result=containers,
        ),
        RuntimeTestCase(
            name="dependency_wiring_configuration",
            operation=RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION,
        ),
        RuntimeTestCase(
            name="dependency_wiring_factories",
            operation=RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES,
        ),
        RuntimeTestCase(
            name="dependency_wiring_automation",
            operation=RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION,
        ),
        RuntimeTestCase(
            name="service_runtime_automation",
            operation=RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION,
        ),
        RuntimeTestCase(
            name="mixins_runtime_automation",
            operation=RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION,
        ),
    ]
    STRUCTLOG_CONFIG_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            name="configure_defaults",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS,
            test_input=None,
            expected_result=None,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_custom_level",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL,
            test_input=None,
            expected_result=None,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_json_renderer",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER,
            test_input=None,
            expected_result=None,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_additional_processors",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS,
            test_input=None,
            expected_result=None,
            should_reset_config=True,
        ),
        RuntimeTestCase(
            name="configure_idempotent",
            operation=RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT,
            test_input=None,
            expected_result=None,
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


class TestFlextRuntime:
    """Unified test suite for FlextRuntime using FlextTestsUtilities."""

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.DICT_LIKE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_dict_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test dict-like object validation.

        Business Rule: is_dict_like accepts object compatible objects.
        test_case.test_input may be None or various types, so we cast to object
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_dict_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIST_LIKE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_list_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test list-like object validation.

        Business Rule: is_list_like accepts object compatible objects.
        test_case.test_input may be None or various types, so we cast to object
        for type compatibility while preserving runtime behavior.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_list_like(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.JSON_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_json_validation(self, test_case: RuntimeTestCase) -> None:
        """Test JSON string validation.

        Business Rule: None is a valid test input - validates that is_valid_json
        correctly returns False for None values.
        """
        tm.that(not isinstance(test_case.test_input, type), eq=True)
        result = FlextRuntime.is_valid_json(cast("t.RuntimeData", test_case.test_input))
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.IDENTIFIER_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_identifier_validation(self, test_case: RuntimeTestCase) -> None:
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
        "test_case",
        RuntimeScenarios.SERIALIZATION_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_safe_get_attribute(self, test_case: RuntimeTestCase) -> None:
        """Test safe attribute retrieval."""
        if test_case.operation == RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS:

            class TestObj:
                attr = "value"

            result = FlextRuntime.safe_get_attribute(TestObj, "attr")
            tm.that(result == "value", eq=True)
        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT
        ):

            class TestObjDefault:
                pass

            result = FlextRuntime.safe_get_attribute(
                TestObjDefault,
                "missing",
                "default",
            )
            tm.that(result == "default", eq=True)
        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT
        ):

            class TestObjNoDefault:
                pass

            result = FlextRuntime.safe_get_attribute(TestObjNoDefault, "missing")
            tm.that(result is None, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.GENERIC_ARGS_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_extract_generic_args(self, test_case: RuntimeTestCase) -> None:
        """Test extraction of generic type arguments.

        Business Rule: extract_generic_args accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        args = FlextRuntime.extract_generic_args(test_input_typed)
        tm.that(args == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.SEQUENCE_TYPE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_sequence_type_detection(self, test_case: RuntimeTestCase) -> None:
        """Test sequence type detection.

        Business Rule: is_sequence_type accepts TypeHintSpecifier compatible objects.
        test_case.test_input may be None or various types, so we cast to TypeHintSpecifier
        for type compatibility while preserving runtime behavior.
        """
        test_input_typed = cast("t.TypeHintSpecifier", test_case.test_input)
        result = FlextRuntime.is_sequence_type(test_input_typed)
        tm.that(result == test_case.expected_result, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIBRARY_ACCESS_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_external_library_access(self, test_case: RuntimeTestCase) -> None:
        """Test external library access."""
        if test_case.operation == RuntimeOperationType.STRUCTLOG_MODULE:
            module = FlextRuntime.structlog()
            tm.that(module is structlog, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["get_logger", "configure"]),
                eq=True,
            )
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE:
            module = FlextRuntime.dependency_providers()
            tm.that(module is providers, eq=True)
            tm.that(
                all(hasattr(module, attr) for attr in ["Singleton", "Factory"]), eq=True
            )
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE:
            module = FlextRuntime.dependency_containers()
            tm.that(module is containers, eq=True)
            tm.that(hasattr(module, "DeclarativeContainer"), eq=True)
        elif (
            test_case.operation == RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION
        ):
            di_container = FlextRuntime.DependencyIntegration.create_container()
            config_provider = FlextRuntime.DependencyIntegration.bind_configuration(
                di_container,
                t.ConfigMap(root={"database": {"dsn": "sqlite://"}}),
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
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES:
            di_container = FlextRuntime.DependencyIntegration.create_container()
            factory_provider = FlextRuntime.DependencyIntegration.register_factory(
                di_container,
                "token_factory",
                lambda: {"token": "abc123"},
            )
            object_provider = FlextRuntime.DependencyIntegration.register_object(
                di_container,
                "static_value",
                42,
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
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION:
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
        elif test_case.operation == RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION:
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
        elif test_case.operation == RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION:

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
                config_type=None,
                config_overrides=None,
                initial_context=None,
            )
            runtime_first = component._get_runtime()
            runtime_second = component._get_runtime()
            tm.that(runtime_first is runtime_second, eq=True)
            tm.that(component.config.app_name == "runtime-aware", eq=True)
            tm.that(component.context is runtime_first.context, eq=True)
            service_result = component.container.get(
                "preseed"
            )
            tm.that(service_result.is_success, eq=True)
            tm.that(
                service_result.value == t.ConfigMap(root={"enabled": True}), eq=True
            )
            factory_result = component.container.get(
                "counter"
            )
            tm.that(factory_result.is_success, eq=True)
            tm.that(factory_result.value == {"count": 1}, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.STRUCTLOG_CONFIG_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_structlog_configuration(self, test_case: RuntimeTestCase) -> None:
        """Test structlog configuration."""
        if test_case.should_reset_config:
            RuntimeScenarios.reset_structlog_config()
        if test_case.operation == RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS:
            FlextRuntime.configure_structlog()
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL
        ):
            FlextRuntime.configure_structlog(log_level=logging.DEBUG)
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER
        ):
            FlextRuntime.configure_structlog(console_renderer=False)
            tm.that(FlextRuntime._structlog_configured is True, eq=True)
        elif (
            test_case.operation
            == RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS
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
        elif test_case.operation == RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT:
            FlextRuntime.configure_structlog()
            FlextRuntime.configure_structlog()
            tm.that(FlextRuntime._structlog_configured is True, eq=True)

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.INTEGRATION_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_runtime_integration(self, test_case: RuntimeTestCase) -> None:
        """Test FlextRuntime integration scenarios."""
        if test_case.operation == RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS:
            tm.that(hasattr(c.Platform, "PATTERN_PHONE_NUMBER"), eq=True)
            tm.that(FlextRuntime.is_valid_json('{"phone": "+5511987654321"}'), eq=True)
        elif test_case.operation == RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY:
            tm.that(hasattr(c.Platform, "PATTERN_EMAIL"), eq=True)
        elif (
            test_case.operation == RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution("database", resolved=True)
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation == RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution(
                "cache",
                resolved=False,
                error_message="Connection refused",
            )
            tm.that(
                FlextContext.Correlation.get_correlation_id() == correlation_id, eq=True
            )
        elif (
            test_case.operation
            == RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE
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
            == RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE
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
            == RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL
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
            == RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL
        ):
            FlextRuntime.configure_structlog()
            FlextRuntime.Integration.setup_service_infrastructure(
                service_name="minimal-service",
                enable_context_correlation=True,
            )
            tm.that(
                FlextContext.Variables.ServiceName.get() == "minimal-service", eq=True
            )
            tm.that(FlextContext.Correlation.get_correlation_id() is not None, eq=True)
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
            tm.that(
                FlextContext.Variables.ServiceName.get() == "no-correlation-service",
                eq=True,
            )
            tm.that(FlextContext.Correlation.get_correlation_id() is None, eq=True)


__all__ = ["TestFlextRuntime"]
