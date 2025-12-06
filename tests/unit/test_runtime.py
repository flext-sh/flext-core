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

import dataclasses
import logging
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from types import ModuleType
from typing import ClassVar, cast

import pytest
import structlog
from dependency_injector import containers, providers

from flext_core import FlextContainer, FlextContext, FlextRuntime, c, r, t
from flext_core.mixins import FlextMixins


class RuntimeOperationType(StrEnum):
    """Runtime operation types for parametrized testing."""

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


@dataclasses.dataclass(frozen=True, slots=True)
class RuntimeTestCase:
    """Runtime test case definition with parametrization data."""

    name: str
    operation: RuntimeOperationType
    # Business Rule: test_input supports both values and types for comprehensive testing
    # GeneralValueType | type[object] | None allows testing runtime type checking with various inputs
    test_input: t.GeneralValueType | type[object] | None = None
    expected_result: bool | tuple[object, ...] | object = None
    should_reset_config: bool = False


class RuntimeScenarios:
    """Centralized runtime test scenarios using c."""

    PHONE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "phone_valid_international",
            RuntimeOperationType.PHONE_VALID,
            "+5511987654321",
            True,
        ),
        RuntimeTestCase(
            "phone_valid_no_country",
            RuntimeOperationType.PHONE_VALID,
            "5511987654321",
            True,
        ),
        RuntimeTestCase(
            "phone_valid_us_format",
            RuntimeOperationType.PHONE_VALID,
            "+1234567890",
            True,
        ),
        RuntimeTestCase(
            "phone_valid_15_digits",
            RuntimeOperationType.PHONE_VALID,
            "123456789012345",
            True,
        ),
        RuntimeTestCase(
            "phone_invalid_too_short",
            RuntimeOperationType.PHONE_INVALID,
            "123",
            False,
        ),
        RuntimeTestCase(
            "phone_invalid_letters",
            RuntimeOperationType.PHONE_INVALID,
            "abc1234567890",
            False,
        ),
        RuntimeTestCase(
            "phone_invalid_country_letters",
            RuntimeOperationType.PHONE_INVALID,
            "+abc123",
            False,
        ),
        RuntimeTestCase(
            "phone_invalid_empty",
            RuntimeOperationType.PHONE_INVALID,
            "",
            False,
        ),
        RuntimeTestCase(
            "phone_non_string_int",
            RuntimeOperationType.PHONE_NON_STRING,
            5511987654321,
            False,
        ),
        RuntimeTestCase(
            "phone_non_string_none",
            RuntimeOperationType.PHONE_NON_STRING,
            None,
            False,
        ),
    ]

    DICT_LIKE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "dict_like_empty",
            RuntimeOperationType.DICT_LIKE_VALID,
            {},
            True,
        ),
        RuntimeTestCase(
            "dict_like_single_key",
            RuntimeOperationType.DICT_LIKE_VALID,
            {"key": "value"},
            True,
        ),
        RuntimeTestCase(
            "dict_like_nested",
            RuntimeOperationType.DICT_LIKE_VALID,
            {"nested": {"dict": True}},
            True,
        ),
        RuntimeTestCase(
            "dict_like_invalid_list",
            RuntimeOperationType.DICT_LIKE_INVALID,
            [],
            False,
        ),
        RuntimeTestCase(
            "dict_like_invalid_string",
            RuntimeOperationType.DICT_LIKE_INVALID,
            "string",
            False,
        ),
        RuntimeTestCase(
            "dict_like_invalid_int",
            RuntimeOperationType.DICT_LIKE_INVALID,
            123,
            False,
        ),
        RuntimeTestCase(
            "dict_like_invalid_none",
            RuntimeOperationType.DICT_LIKE_INVALID,
            None,
            False,
        ),
    ]

    LIST_LIKE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "list_like_empty",
            RuntimeOperationType.LIST_LIKE_VALID,
            [],
            True,
        ),
        RuntimeTestCase(
            "list_like_integers",
            RuntimeOperationType.LIST_LIKE_VALID,
            [1, 2, 3],
            True,
        ),
        RuntimeTestCase(
            "list_like_strings",
            RuntimeOperationType.LIST_LIKE_VALID,
            ["a", "b", "c"],
            True,
        ),
        RuntimeTestCase(
            "list_like_invalid_dict",
            RuntimeOperationType.LIST_LIKE_INVALID,
            {},
            False,
        ),
        RuntimeTestCase(
            "list_like_invalid_string",
            RuntimeOperationType.LIST_LIKE_INVALID,
            "string",
            False,
        ),
        RuntimeTestCase(
            "list_like_invalid_int",
            RuntimeOperationType.LIST_LIKE_INVALID,
            123,
            False,
        ),
        RuntimeTestCase(
            "list_like_invalid_none",
            RuntimeOperationType.LIST_LIKE_INVALID,
            None,
            False,
        ),
    ]

    JSON_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "json_valid_object",
            RuntimeOperationType.JSON_VALID,
            '{"key": "value"}',
            True,
        ),
        RuntimeTestCase(
            "json_valid_empty_array",
            RuntimeOperationType.JSON_VALID,
            "[]",
            True,
        ),
        RuntimeTestCase(
            "json_valid_array",
            RuntimeOperationType.JSON_VALID,
            "[1, 2, 3]",
            True,
        ),
        RuntimeTestCase(
            "json_valid_string",
            RuntimeOperationType.JSON_VALID,
            '"string"',
            True,
        ),
        RuntimeTestCase(
            "json_valid_null",
            RuntimeOperationType.JSON_VALID,
            "null",
            True,
        ),
        RuntimeTestCase(
            "json_invalid_plain_text",
            RuntimeOperationType.JSON_INVALID,
            "not json",
            False,
        ),
        RuntimeTestCase(
            "json_invalid_malformed",
            RuntimeOperationType.JSON_INVALID,
            "{invalid}",
            False,
        ),
        RuntimeTestCase(
            "json_invalid_empty",
            RuntimeOperationType.JSON_INVALID,
            "",
            False,
        ),
        RuntimeTestCase(
            "json_non_string_dict",
            RuntimeOperationType.JSON_NON_STRING,
            {"key": "value"},
            False,
        ),
        RuntimeTestCase(
            "json_non_string_list",
            RuntimeOperationType.JSON_NON_STRING,
            [1, 2, 3],
            False,
        ),
        RuntimeTestCase(
            "json_non_string_none",
            RuntimeOperationType.JSON_NON_STRING,
            None,
            False,
        ),
    ]

    IDENTIFIER_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "identifier_valid_lowercase",
            RuntimeOperationType.IDENTIFIER_VALID,
            "variable",
            True,
        ),
        RuntimeTestCase(
            "identifier_valid_private",
            RuntimeOperationType.IDENTIFIER_VALID,
            "_private",
            True,
        ),
        RuntimeTestCase(
            "identifier_valid_class_name",
            RuntimeOperationType.IDENTIFIER_VALID,
            "ClassName",
            True,
        ),
        RuntimeTestCase(
            "identifier_valid_snake_case",
            RuntimeOperationType.IDENTIFIER_VALID,
            "function_name",
            True,
        ),
        RuntimeTestCase(
            "identifier_invalid_starts_with_digit",
            RuntimeOperationType.IDENTIFIER_INVALID,
            "123invalid",
            False,
        ),
        RuntimeTestCase(
            "identifier_invalid_hyphen",
            RuntimeOperationType.IDENTIFIER_INVALID,
            "invalid-name",
            False,
        ),
        RuntimeTestCase(
            "identifier_invalid_space",
            RuntimeOperationType.IDENTIFIER_INVALID,
            "invalid name",
            False,
        ),
        RuntimeTestCase(
            "identifier_invalid_empty",
            RuntimeOperationType.IDENTIFIER_INVALID,
            "",
            False,
        ),
        RuntimeTestCase(
            "identifier_non_string_int",
            RuntimeOperationType.IDENTIFIER_NON_STRING,
            123,
            False,
        ),
        RuntimeTestCase(
            "identifier_non_string_none",
            RuntimeOperationType.IDENTIFIER_NON_STRING,
            None,
            False,
        ),
    ]

    GENERIC_ARGS_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "extract_generic_list",
            RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
            list[str],
            (str,),
        ),
        RuntimeTestCase(
            "extract_generic_dict",
            RuntimeOperationType.EXTRACT_GENERIC_GENERIC_TYPE,
            dict[str, int],
            (str, int),
        ),
        RuntimeTestCase(
            "extract_generic_non_generic_str",
            RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
            str,
            (),
        ),
        RuntimeTestCase(
            "extract_generic_non_generic_int",
            RuntimeOperationType.EXTRACT_GENERIC_NON_GENERIC,
            int,
            (),
        ),
        RuntimeTestCase(
            "extract_generic_exception_none",
            RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
            None,
            (),
        ),
        RuntimeTestCase(
            "extract_generic_exception_string",
            RuntimeOperationType.EXTRACT_GENERIC_EXCEPTION,
            "not a type",
            (),
        ),
    ]

    SEQUENCE_TYPE_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "sequence_type_list_of_str",
            RuntimeOperationType.SEQUENCE_TYPE_VALID,
            list[str],
            True,
        ),
        RuntimeTestCase(
            "sequence_type_tuple",
            RuntimeOperationType.SEQUENCE_TYPE_VALID,
            tuple[int, ...],
            True,
        ),
        RuntimeTestCase(
            "sequence_type_str_is_sequence",
            RuntimeOperationType.SEQUENCE_TYPE_VALID,
            str,
            True,
        ),
        RuntimeTestCase(
            "sequence_type_invalid_dict",
            RuntimeOperationType.SEQUENCE_TYPE_INVALID,
            dict[str, int],
            False,
        ),
        RuntimeTestCase(
            "sequence_type_invalid_int",
            RuntimeOperationType.SEQUENCE_TYPE_INVALID,
            int,
            False,
        ),
        RuntimeTestCase(
            "sequence_type_exception_none",
            RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
            None,
            False,
        ),
        RuntimeTestCase(
            "sequence_type_exception_string",
            RuntimeOperationType.SEQUENCE_TYPE_EXCEPTION,
            "not a type",
            False,
        ),
    ]

    SERIALIZATION_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "safe_get_attribute_exists",
            RuntimeOperationType.SAFE_GET_ATTRIBUTE_EXISTS,
            None,
            "value",
        ),
        RuntimeTestCase(
            "safe_get_attribute_missing_with_default",
            RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT,
            None,
            "default",
        ),
        RuntimeTestCase(
            "safe_get_attribute_missing_no_default",
            RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT,
            None,
            None,
        ),
    ]

    LIBRARY_ACCESS_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "structlog_module",
            RuntimeOperationType.STRUCTLOG_MODULE,
            None,
            structlog,
        ),
        RuntimeTestCase(
            "dependency_providers",
            RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE,
            None,
            providers,
        ),
        RuntimeTestCase(
            "dependency_containers",
            RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE,
            None,
            containers,
        ),
        RuntimeTestCase(
            "dependency_wiring_configuration",
            RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION,
        ),
        RuntimeTestCase(
            "dependency_wiring_factories",
            RuntimeOperationType.DEPENDENCY_WIRING_FACTORIES,
        ),
        RuntimeTestCase(
            "dependency_wiring_automation",
            RuntimeOperationType.DEPENDENCY_WIRING_AUTOMATION,
        ),
        RuntimeTestCase(
            "service_runtime_automation",
            RuntimeOperationType.SERVICE_RUNTIME_AUTOMATION,
        ),
        RuntimeTestCase(
            "mixins_runtime_automation",
            RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION,
        ),
    ]

    STRUCTLOG_CONFIG_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "configure_defaults",
            RuntimeOperationType.CONFIGURE_STRUCTLOG_DEFAULTS,
            None,
            None,
            True,
        ),
        RuntimeTestCase(
            "configure_custom_level",
            RuntimeOperationType.CONFIGURE_STRUCTLOG_CUSTOM_LOG_LEVEL,
            None,
            None,
            True,
        ),
        RuntimeTestCase(
            "configure_json_renderer",
            RuntimeOperationType.CONFIGURE_STRUCTLOG_JSON_RENDERER,
            None,
            None,
            True,
        ),
        RuntimeTestCase(
            "configure_additional_processors",
            RuntimeOperationType.CONFIGURE_STRUCTLOG_ADDITIONAL_PROCESSORS,
            None,
            None,
            True,
        ),
        RuntimeTestCase(
            "configure_idempotent",
            RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT,
            None,
            None,
            True,
        ),
    ]

    INTEGRATION_SCENARIOS: ClassVar[list[RuntimeTestCase]] = [
        RuntimeTestCase(
            "constants_patterns",
            RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS,
        ),
        RuntimeTestCase(
            "layer_hierarchy",
            RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY,
        ),
        RuntimeTestCase(
            "track_service_resolution_success",
            RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS,
        ),
        RuntimeTestCase(
            "track_service_resolution_failure",
            RuntimeOperationType.TRACK_SERVICE_RESOLUTION_FAILURE,
        ),
        RuntimeTestCase(
            "track_domain_event_with_aggregate",
            RuntimeOperationType.TRACK_DOMAIN_EVENT_WITH_AGGREGATE,
        ),
        RuntimeTestCase(
            "track_domain_event_without_aggregate",
            RuntimeOperationType.TRACK_DOMAIN_EVENT_WITHOUT_AGGREGATE,
        ),
        RuntimeTestCase(
            "setup_service_infrastructure_full",
            RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_FULL,
        ),
        RuntimeTestCase(
            "setup_service_infrastructure_minimal",
            RuntimeOperationType.SETUP_SERVICE_INFRASTRUCTURE_MINIMAL,
        ),
        RuntimeTestCase(
            "setup_service_without_correlation",
            RuntimeOperationType.SETUP_SERVICE_WITHOUT_CORRELATION,
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
        RuntimeScenarios.PHONE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_phone_validation(self, test_case: RuntimeTestCase) -> None:
        """Test phone number validation.

        Business Rule: None is a valid test input - validates that is_valid_phone
        correctly returns False for None values.
        """
        result = FlextRuntime.is_valid_phone(
            cast("t.GeneralValueType", test_case.test_input),
        )
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.DICT_LIKE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_dict_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test dict-like object validation.

        Business Rule: is_dict_like accepts GeneralValueType compatible objects.
        test_case.test_input may be None or various types, so we cast to GeneralValueType
        for type compatibility while preserving runtime behavior.
        """
        # Business Rule: Cast to GeneralValueType for type compatibility
        # None and various types are compatible with GeneralValueType at runtime
        test_input_typed = cast("t.GeneralValueType", test_case.test_input)
        result = FlextRuntime.is_dict_like(test_input_typed)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIST_LIKE_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_list_like_validation(self, test_case: RuntimeTestCase) -> None:
        """Test list-like object validation.

        Business Rule: is_list_like accepts GeneralValueType compatible objects.
        test_case.test_input may be None or various types, so we cast to GeneralValueType
        for type compatibility while preserving runtime behavior.
        """
        # Business Rule: Cast to GeneralValueType for type compatibility
        # None and various types are compatible with GeneralValueType at runtime
        test_input_typed = cast("t.GeneralValueType", test_case.test_input)
        result = FlextRuntime.is_list_like(test_input_typed)
        assert result == test_case.expected_result

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
        result = FlextRuntime.is_valid_json(
            cast("t.GeneralValueType", test_case.test_input),
        )
        assert result == test_case.expected_result

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
        result = FlextRuntime.is_valid_identifier(
            cast("t.GeneralValueType", test_case.test_input),
        )
        assert result == test_case.expected_result

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

            test_obj = TestObj()
            # Type narrowing: TestObj is compatible with GeneralValueType
            test_obj_cast: t.GeneralValueType = cast("t.GeneralValueType", test_obj)
            result = FlextRuntime.safe_get_attribute(test_obj_cast, "attr")
            assert result == "value"
        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_DEFAULT
        ):

            class TestObjDefault:
                pass

            test_obj_default_obj = TestObjDefault()
            # Type narrowing: TestObjDefault is compatible with GeneralValueType
            test_obj_default_cast: t.GeneralValueType = cast(
                "t.GeneralValueType",
                test_obj_default_obj,
            )
            result = FlextRuntime.safe_get_attribute(
                test_obj_default_cast,
                "missing",
                "default",
            )
            assert result == "default"
        elif (
            test_case.operation
            == RuntimeOperationType.SAFE_GET_ATTRIBUTE_MISSING_NO_DEFAULT
        ):

            class TestObjNoDefault:
                pass

            # Business Rule: TestObjNoDefault instances are compatible with GeneralValueType at runtime
            # Cast to GeneralValueType for type compatibility
            test_obj_no_default = cast("t.GeneralValueType", TestObjNoDefault())
            result = FlextRuntime.safe_get_attribute(test_obj_no_default, "missing")
            assert result is None

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
        # Business Rule: Cast to TypeHintSpecifier for type compatibility
        # None and various types are compatible with TypeHintSpecifier at runtime
        test_input_typed = cast("t.Utility.TypeHintSpecifier", test_case.test_input)
        args = FlextRuntime.extract_generic_args(test_input_typed)
        assert args == test_case.expected_result

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
        # Business Rule: Cast to TypeHintSpecifier for type compatibility
        # None and various types are compatible with TypeHintSpecifier at runtime
        test_input_typed = cast("t.Utility.TypeHintSpecifier", test_case.test_input)
        result = FlextRuntime.is_sequence_type(test_input_typed)
        assert result == test_case.expected_result

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.LIBRARY_ACCESS_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_external_library_access(self, test_case: RuntimeTestCase) -> None:
        """Test external library access."""
        if test_case.operation == RuntimeOperationType.STRUCTLOG_MODULE:
            module = FlextRuntime.structlog()
            assert module is structlog
            assert all(hasattr(module, attr) for attr in ["get_logger", "configure"])
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_PROVIDERS_MODULE:
            module = FlextRuntime.dependency_providers()
            assert module is providers
            assert all(hasattr(module, attr) for attr in ["Singleton", "Factory"])
        elif test_case.operation == RuntimeOperationType.DEPENDENCY_CONTAINERS_MODULE:
            module = FlextRuntime.dependency_containers()
            assert module is containers
            assert hasattr(module, "DeclarativeContainer")
        elif (
            test_case.operation == RuntimeOperationType.DEPENDENCY_WIRING_CONFIGURATION
        ):
            di_container = FlextRuntime.DependencyIntegration.create_container()
            config_provider = FlextRuntime.DependencyIntegration.bind_configuration(
                di_container,
                {"database": {"dsn": "sqlite://"}},
            )
            assert isinstance(config_provider, providers.Configuration)
            # Type narrowing: di_container.config is providers.Configuration
            # Access nested attributes via getattr for mypy compatibility
            config = di_container.config
            database_config = getattr(config, "database", None)
            assert database_config is not None
            dsn_value = database_config.dsn()
            assert dsn_value == "sqlite://"

            module = ModuleType("di_config_module")

            @FlextRuntime.DependencyIntegration.inject
            def read_config(
                dsn: str = FlextRuntime.DependencyIntegration.Provide[
                    "config.database.dsn"
                ],
            ) -> str:
                return dsn

            # Type annotation for dynamic module attribute
            setattr(module, "read_config", read_config)
            di_container.wire(modules=[module])
            try:
                # Type narrowing: module has read_config attribute after setattr
                # Mypy limitation: can't infer dynamic module attributes
                read_func = getattr(module, "read_config")  # type: ignore[attr-defined]
                assert callable(read_func)
                result = read_func()
                assert result == "sqlite://"
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
            assert isinstance(factory_provider, providers.Singleton)
            assert factory_provider() == {"token": "abc123"}
            assert object_provider() == 42

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
                return token, static

            # Type annotation for dynamic module attribute
            setattr(module, "consume", consume)
            di_container.wire(modules=[module])
            try:
                # Type narrowing: module has consume attribute after setattr
                # Mypy limitation: can't infer dynamic module attributes
                consume_func = getattr(module, "consume")  # type: ignore[attr-defined]
                assert callable(consume_func)
                tokens, value = consume_func()
                assert tokens == {"token": "abc123"}
                assert value == 42
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
                return static_value, token, config_flag, resource

            # Type annotation for dynamic module attribute
            setattr(module, "consume", consume_automation)

            di_container = FlextRuntime.DependencyIntegration.create_container(
                config={"flags": {"enabled": True}},
                services={"static_value": 7},
                factories={"token_factory": token_factory},
                resources={"api_client": lambda: {"connected": True}},
                wire_modules=[module],
                factory_cache=False,
            )

            try:
                # Type narrowing: module has consume attribute after setattr
                # Mypy limitation: can't infer dynamic module attributes
                consume_func = getattr(module, "consume")  # type: ignore[attr-defined]
                assert callable(consume_func)
                first_static, first_token, config_enabled, resource_value = (
                    consume_func()
                )
                second_static, second_token, _, _ = consume_func()

                assert first_static == second_static == 7
                assert config_enabled is True
                assert resource_value == {"connected": True}
                assert first_token["token"] == 1
                assert second_token["token"] == 2
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
                return flag, token, resource

            # Type annotation for dynamic module attribute
            setattr(module, "consume", consume_service)

            runtime = FlextRuntime.create_service_runtime(
                config_overrides={"app_name": "runtime-service"},
                services={"feature_flag": True},
                factories={"token_factory": token_factory},
                resources={"api_client": lambda: {"connected": True}},
                wire_modules=[module],
            )

            try:
                # Type narrowing: module has consume attribute after setattr
                # Mypy limitation: can't infer dynamic module attributes
                consume_func = getattr(module, "consume")  # type: ignore[attr-defined]
                assert callable(consume_func)
                feature_flag, first_token, resource = consume_func()
                _, second_token, _ = consume_func()

                assert runtime.config.app_name == "runtime-service"
                assert feature_flag is True
                assert resource == {"connected": True}
                assert first_token["count"] == 1
                assert second_token["count"] == 2
            finally:
                # Type narrowing: runtime.container is p.Container.DI protocol
                # Cast to FlextContainer to access private _di_bridge attribute
                container = cast("FlextContainer", runtime.container)
                container._di_bridge.unwire()
        elif test_case.operation == RuntimeOperationType.MIXINS_RUNTIME_AUTOMATION:

            class RuntimeAwareComponent(FlextMixins):
                @classmethod
                def _runtime_bootstrap_options(cls) -> t.Types.RuntimeBootstrapOptions:
                    # factories should be Mapping[str, Callable[[], ScalarValue | Sequence | Mapping]]
                    # RuntimeBootstrapOptions["factories"] has the correct type
                    def counter_factory() -> t.GeneralValueType:
                        return {"count": 1}

                    # Type: factories expects Callable[[], ScalarValue | Sequence | Mapping]
                    # counter_factory returns dict[str, int] which is Mapping[str, ScalarValue]
                    # Cast to satisfy type checker
                    counter_factory_typed: Callable[
                        [],
                        (
                            t.ScalarValue
                            | Sequence[t.ScalarValue]
                            | Mapping[str, t.ScalarValue]
                        ),
                    ] = cast(
                        "Callable[[], t.ScalarValue | Sequence[t.ScalarValue] | Mapping[str, t.ScalarValue]]",
                        counter_factory,
                    )
                    factories_dict: Mapping[
                        str,
                        Callable[
                            [],
                            (
                                t.ScalarValue
                                | Sequence[t.ScalarValue]
                                | Mapping[str, t.ScalarValue]
                            ),
                        ],
                    ] = {
                        "counter": counter_factory_typed,
                    }
                    return {
                        "config_overrides": {"app_name": "runtime-aware"},
                        "services": {"preseed": {"enabled": True}},
                        "factories": factories_dict,
                    }

            component = RuntimeAwareComponent()

            runtime_first = component._get_runtime()
            runtime_second = component._get_runtime()

            assert runtime_first is runtime_second
            assert component.config.app_name == "runtime-aware"
            assert component.context is runtime_first.context

            # Type parameter must be explicit for mypy inference
            # Mypy limitation: generic method syntax get[T]() not fully supported
            # Call method directly and let runtime type inference work
            # Mypy infers Result[Never] for generic methods without explicit type parameter
            # Annotate explicitly to help mypy
            service_result_raw: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]", component.container.get("preseed")
            )
            # Type narrowing: container.get returns r[T], cast to expected type
            service_result: r[t.GeneralValueType] = service_result_raw
            assert service_result.is_success
            assert service_result.value == {"enabled": True}

            # Type parameter must be explicit for mypy inference
            # Mypy limitation: generic method syntax get[T]() not fully supported
            # Call method directly and let runtime type inference work
            # Mypy infers Result[Never] for generic methods without explicit type parameter
            # Annotate explicitly to help mypy
            factory_result_raw: r[t.GeneralValueType] = cast(
                "r[t.GeneralValueType]", component.container.get("counter")
            )
            # Type narrowing: container.get returns r[T], cast to expected type
            factory_result: r[t.GeneralValueType] = factory_result_raw
            assert factory_result.is_success
            assert factory_result.value == {"count": 1}

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
            assert FlextRuntime._structlog_configured is True
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

            # Business Rule: Callable processors are compatible with GeneralValueType at runtime
            # structlog accepts callable processors for custom processing
            processor_typed: t.GeneralValueType = cast(
                "t.GeneralValueType",
                custom_processor,
            )
            FlextRuntime.configure_structlog(additional_processors=[processor_typed])
            assert FlextRuntime._structlog_configured is True
        elif test_case.operation == RuntimeOperationType.CONFIGURE_STRUCTLOG_IDEMPOTENT:
            FlextRuntime.configure_structlog()
            FlextRuntime.configure_structlog()
            assert FlextRuntime._structlog_configured is True

    @pytest.mark.parametrize(
        "test_case",
        RuntimeScenarios.INTEGRATION_SCENARIOS,
        ids=lambda c: c.name,
    )
    def test_runtime_integration(self, test_case: RuntimeTestCase) -> None:
        """Test FlextRuntime integration scenarios."""
        if test_case.operation == RuntimeOperationType.INTEGRATION_CONSTANTS_PATTERNS:
            assert hasattr(c.Platform, "PATTERN_PHONE_NUMBER")
            assert FlextRuntime.is_valid_phone("+5511987654321")
        elif test_case.operation == RuntimeOperationType.INTEGRATION_LAYER_HIERARCHY:
            assert c is not None and FlextRuntime is not None
            assert c.Platform.PATTERN_EMAIL is not None
        elif (
            test_case.operation == RuntimeOperationType.TRACK_SERVICE_RESOLUTION_SUCCESS
        ):
            FlextRuntime.configure_structlog()
            correlation_id = FlextContext.Utilities.ensure_correlation_id()
            FlextRuntime.Integration.track_service_resolution("database", resolved=True)
            assert FlextContext.Correlation.get_correlation_id() == correlation_id
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
            assert FlextContext.Correlation.get_correlation_id() is not None
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
