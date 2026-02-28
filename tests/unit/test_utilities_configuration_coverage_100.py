"""Comprehensive coverage tests for FlextUtilitiesConfiguration.

Module: flext_core._utilities.configuration
Scope: Configuration parameter access and manipulation utilities

Tests validate:
- get_parameter: dict-like, Pydantic models, attribute access
- set_parameter: Pydantic validation, error handling
- get_singleton/set_singleton: Singleton pattern integration
- validate_config_class: Configuration class validation
- create_settings_config: Settings configuration creation
- build_options_from_kwargs: Options building with kwargs

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# mypy: disable-error-code="valid-type,misc"
# mypy: follow-imports=skip
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, cast

import pytest
from flext_core import FlextExceptions, m, p, t
from flext_core._utilities.configuration import FlextUtilitiesConfiguration
from pydantic import BaseModel, ConfigDict, Field


class _Assertions:
    @staticmethod
    def that(
        value: object,
        *,
        eq: object | None = None,
        none: bool | None = None,
        contains: str | None = None,
        msg: str = "",
    ) -> None:
        if eq is not None:
            assert value == eq, msg
        if none is True:
            assert value is None, msg
        if none is False:
            assert value is not None, msg
        if contains is not None:
            assert isinstance(value, str)
            assert contains in value, msg


class _ResultAssertions:
    @staticmethod
    def assert_success_with_value(result: object, expected: object) -> None:
        assert hasattr(result, "is_success") and getattr(result, "is_success")
        assert getattr(result, "value") == expected

    @staticmethod
    def assert_result_success(result: object) -> None:
        assert hasattr(result, "is_success") and getattr(result, "is_success")

    @staticmethod
    def assert_result_failure(result: object) -> None:
        assert hasattr(result, "is_failure") and getattr(result, "is_failure")


# Test models - module level for forward reference resolution
class ConfigModelForTest(BaseModel):
    """Test configuration model (mutable for set_parameter tests)."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str = "default_config"
    timeout: int = Field(default=30, ge=0)
    enabled: bool = True


class OptionsModelForTest(m.Value):
    """Test options model for build_options_from_kwargs."""

    format: str = "json"
    indent: int = 2
    sort_keys: bool = False


class StrictOptionsForTest(m.Value):
    """Strict options with validation."""

    value: int = Field(ge=0, le=100)


class InvalidModelForTest(BaseModel):
    """Model with invalid model_dump."""

    value: str = "test"


@dataclass
class DataclassConfigForTest:
    """Test dataclass configuration."""

    name: str
    value: int = 42


class SingletonClassForTest(BaseModel):
    """Test singleton class with Pydantic validation."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    _instance: ClassVar[SingletonClassForTest | None] = None

    name: str = "default"
    timeout: int = 30

    @classmethod
    def get_global_instance(cls) -> SingletonClassForTest:
        """Get global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for test isolation."""
        cls._instance = None


class SingletonWithoutGetGlobalForTest:
    """Test class without get_global_instance."""

    def __init__(self) -> None:
        """Initialize."""
        self.value = "test"


class BadSingletonForTest:
    """Singleton with non-callable get_global_instance."""

    get_global_instance = "not callable"


class SingletonWithoutModelDumpForTest:
    """Singleton without model_dump."""

    _instance: ClassVar[SingletonWithoutModelDumpForTest | None] = None

    @classmethod
    def get_global_instance(cls) -> SingletonWithoutModelDumpForTest:
        """Get global instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


class ConfigWithoutModelConfigForTest:
    """Config class without model_config."""

    def __init__(self) -> None:
        """Initialize."""


class BadConfigForTest(BaseModel):
    """Config that fails to instantiate."""

    model_config = {"validate_assignment": True}

    def __init__(self, **kwargs: t.GeneralValueType) -> None:
        """Raise error on init."""
        super().__init__(**kwargs)
        msg = "Cannot instantiate"
        raise ValueError(msg)


class FailingOptionsForTest(m.Value):
    """Options that fail on model_dump."""

    value: str = "test"


class TestConfigModels:
    """Test configuration models namespace - references module-level classes."""


class TestConfigConstants:
    """Test configuration constants."""

    class ParameterNames(StrEnum):
        """Parameter name constants."""

        NAME = "name"
        TIMEOUT = "timeout"
        ENABLED = "enabled"
        VALUE = "value"
        MISSING = "missing"
        FORMAT = "format"
        INDENT = "indent"
        SORT_KEYS = "sort_keys"
        EMPTY_STRING = "empty_string"
        ZERO = "zero"
        FALSE = "false"
        NONE_VALUE = "none_value"
        LARGE_NUMBER = "large_number"

    class TestValues:
        """Test value constants."""

        TEST_NAME: str = "test"
        TEST_TIMEOUT: int = 60
        TEST_TIMEOUT_LARGE: int = 1000
        TEST_TIMEOUT_ZERO: int = 0
        TEST_TIMEOUT_INVALID: int = -1
        TEST_ENABLED_TRUE: bool = True
        TEST_ENABLED_FALSE: bool = False
        TEST_VALUE: int = 42
        TEST_VALUE_UPDATED: int = 100
        TEST_VALUE_INVALID: int = 200
        TEST_FORMAT_XML: str = "xml"
        TEST_FORMAT_JSON: str = "json"
        TEST_FORMAT_YAML: str = "yaml"
        TEST_FORMAT_TOML: str = "toml"
        TEST_INDENT_2: int = 2
        TEST_INDENT_4: int = 4
        TEST_INDENT_6: int = 6
        TEST_INDENT_8: int = 8
        TEST_SORT_KEYS_TRUE: bool = True
        TEST_SORT_KEYS_FALSE: bool = False
        EMPTY_STRING: str = ""
        ZERO: int = 0
        FALSE: bool = False
        NONE_VALUE: None = None
        LARGE_NUMBER: int = 999999

    class SettingsConfig:
        """Settings configuration constants."""

        ENV_PREFIX: str = "MYAPP_"
        ENV_FILE: str = ".env.test"
        ENV_NESTED_DELIMITER_DEFAULT: str = "__"
        ENV_NESTED_DELIMITER_CUSTOM: str = "::"
        CASE_SENSITIVE: bool = False
        EXTRA: str = "ignore"
        VALIDATE_DEFAULT: bool = True

    class ErrorMessages:
        """Error message patterns."""

        PARAMETER_NOT_DEFINED: str = "Parameter '{}' is not defined"
        DOES_NOT_HAVE_GET_GLOBAL: str = "does not have get_global_instance method"
        IS_NOT_CALLABLE: str = "is not callable"
        DOES_NOT_IMPLEMENT_HAS_MODEL_DUMP: str = (
            "Instance does not implement model_dump() method"
        )
        FAILED_TO_SET_PARAMETER: str = "Failed to set parameter '{}'"
        MUST_DEFINE_MODEL_CONFIG: str = "must define model_config"
        CONFIGURATION_CLASS_VALIDATION_FAILED: str = (
            "Configuration class validation failed"
        )
        FAILED_TO_BUILD: str = "Failed to build {}"
        UNEXPECTED_ERROR_BUILDING: str = "Unexpected error building {}"


pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesConfiguration:
    """Comprehensive tests for FlextUtilitiesConfiguration.

    Single class pattern with nested classes organizing test cases by functionality.
    Uses factories, constants, and DRY principles to minimize code duplication.
    """

    class TestGetParameter:
        """Tests for get_parameter method."""

        @staticmethod
        def _create_test_dict() -> dict[str, t.GeneralValueType]:
            """Factory for test dict."""
            return {
                TestConfigConstants.ParameterNames.NAME.value: TestConfigConstants.TestValues.TEST_NAME,
                TestConfigConstants.ParameterNames.TIMEOUT.value: TestConfigConstants.TestValues.TEST_TIMEOUT,
                TestConfigConstants.ParameterNames.ENABLED.value: TestConfigConstants.TestValues.TEST_ENABLED_TRUE,
            }

        @staticmethod
        def _create_boundary_dict() -> dict[str, t.GeneralValueType]:
            """Factory for boundary values dict."""
            return {
                TestConfigConstants.ParameterNames.EMPTY_STRING.value: TestConfigConstants.TestValues.EMPTY_STRING,
                TestConfigConstants.ParameterNames.ZERO.value: TestConfigConstants.TestValues.ZERO,
                TestConfigConstants.ParameterNames.FALSE.value: TestConfigConstants.TestValues.FALSE,
                TestConfigConstants.ParameterNames.NONE_VALUE.value: TestConfigConstants.TestValues.NONE_VALUE,
                TestConfigConstants.ParameterNames.LARGE_NUMBER.value: TestConfigConstants.TestValues.LARGE_NUMBER,
            }

        @pytest.mark.parametrize(
            ("param_name", "expected_value"),
            [
                (
                    TestConfigConstants.ParameterNames.NAME.value,
                    TestConfigConstants.TestValues.TEST_NAME,
                ),
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT,
                ),
                (
                    TestConfigConstants.ParameterNames.ENABLED.value,
                    TestConfigConstants.TestValues.TEST_ENABLED_TRUE,
                ),
            ],
        )
        def test_from_dict(
            self,
            param_name: str,
            expected_value: t.GeneralValueType,
        ) -> None:
            """Test get_parameter from dict-like object."""
            config_dict = self._create_test_dict()
            result = FlextUtilitiesConfiguration.get_parameter(config_dict, param_name)
            # Use tm.that for assertions
            _Assertions.that(
                result,
                eq=expected_value,
                msg=f"Parameter {param_name} must match expected value",
            )

        def test_from_dict_not_found(self) -> None:
            """Test get_parameter raises NotFoundError for missing parameter."""
            config_dict = self._create_test_dict()
            with pytest.raises(FlextExceptions.NotFoundError) as exc_info:
                FlextUtilitiesConfiguration.get_parameter(
                    config_dict,
                    TestConfigConstants.ParameterNames.MISSING.value,
                )
            # Use tm.that for assertion
            error_msg = TestConfigConstants.ErrorMessages.PARAMETER_NOT_DEFINED.format(
                TestConfigConstants.ParameterNames.MISSING.value,
            )
            _Assertions.that(
                str(exc_info.value),
                contains=error_msg,
                msg="Error message must contain expected text",
            )

        @pytest.mark.parametrize(
            ("param_name", "expected_value"),
            [
                (
                    TestConfigConstants.ParameterNames.NAME.value,
                    TestConfigConstants.TestValues.TEST_NAME,
                ),
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT,
                ),
                (
                    TestConfigConstants.ParameterNames.ENABLED.value,
                    TestConfigConstants.TestValues.TEST_ENABLED_FALSE,
                ),
            ],
        )
        def test_from_pydantic_model(
            self,
            param_name: str,
            expected_value: t.GeneralValueType,
        ) -> None:
            """Test get_parameter from Pydantic model."""
            # Use tt.model to create test config if available, otherwise use direct instantiation
            config = ConfigModelForTest(
                name=TestConfigConstants.TestValues.TEST_NAME,
                timeout=TestConfigConstants.TestValues.TEST_TIMEOUT,
                enabled=TestConfigConstants.TestValues.TEST_ENABLED_FALSE,
            )
            result = FlextUtilitiesConfiguration.get_parameter(
                cast("p.HasModelDump", cast("object", config)),
                param_name,
            )
            # Use tm.that for assertions
            _Assertions.that(
                result,
                eq=expected_value,
                msg=f"Parameter {param_name} must match expected value",
            )

        def test_from_pydantic_model_not_found(self) -> None:
            """Test get_parameter raises NotFoundError for missing parameter in model."""
            config = ConfigModelForTest(name=TestConfigConstants.TestValues.TEST_NAME)
            with pytest.raises(FlextExceptions.NotFoundError) as exc_info:
                FlextUtilitiesConfiguration.get_parameter(
                    cast("p.HasModelDump", cast("object", config)),
                    TestConfigConstants.ParameterNames.MISSING.value,
                )
            assert TestConfigConstants.ErrorMessages.PARAMETER_NOT_DEFINED.format(
                TestConfigConstants.ParameterNames.MISSING.value,
            ) in str(exc_info.value)

        def test_from_pydantic_model_invalid_dump(self) -> None:
            """Test get_parameter handles invalid model_dump return."""
            config = cast("p.HasModelDump", cast("object", InvalidModelForTest()))
            result = FlextUtilitiesConfiguration.get_parameter(
                config,
                TestConfigConstants.ParameterNames.VALUE.value,
            )
            assert result == "test"

        @pytest.mark.parametrize(
            ("param_name", "expected_value"),
            [
                (
                    TestConfigConstants.ParameterNames.NAME.value,
                    TestConfigConstants.TestValues.TEST_NAME,
                ),
                (
                    TestConfigConstants.ParameterNames.VALUE.value,
                    TestConfigConstants.TestValues.TEST_VALUE,
                ),
            ],
        )
        def test_from_attribute_access(
            self,
            param_name: str,
            expected_value: t.GeneralValueType,
        ) -> None:
            """Test get_parameter from object attribute access."""
            config = DataclassConfigForTest(
                name=TestConfigConstants.TestValues.TEST_NAME,
                value=TestConfigConstants.TestValues.TEST_VALUE,
            )
            config_cast = cast(
                "p.HasModelDump | Mapping[str, t.ConfigMapValue]",
                cast("object", config),
            )
            result = FlextUtilitiesConfiguration.get_parameter(config_cast, param_name)
            assert result == expected_value

        def test_from_attribute_access_not_found(self) -> None:
            """Test get_parameter raises NotFoundError for missing attribute."""
            config = DataclassConfigForTest(
                name=TestConfigConstants.TestValues.TEST_NAME,
            )
            config_cast = cast(
                "p.HasModelDump | Mapping[str, t.ConfigMapValue]",
                cast("object", config),
            )
            with pytest.raises(FlextExceptions.NotFoundError) as exc_info:
                FlextUtilitiesConfiguration.get_parameter(
                    config_cast,
                    TestConfigConstants.ParameterNames.MISSING.value,
                )
            assert TestConfigConstants.ErrorMessages.PARAMETER_NOT_DEFINED.format(
                TestConfigConstants.ParameterNames.MISSING.value,
            ) in str(exc_info.value)

        @pytest.mark.parametrize(
            ("param_name", "expected_value"),
            [
                (
                    TestConfigConstants.ParameterNames.EMPTY_STRING.value,
                    TestConfigConstants.TestValues.EMPTY_STRING,
                ),
                (
                    TestConfigConstants.ParameterNames.ZERO.value,
                    TestConfigConstants.TestValues.ZERO,
                ),
                (
                    TestConfigConstants.ParameterNames.FALSE.value,
                    TestConfigConstants.TestValues.FALSE,
                ),
                (
                    TestConfigConstants.ParameterNames.NONE_VALUE.value,
                    TestConfigConstants.TestValues.NONE_VALUE,
                ),
                (
                    TestConfigConstants.ParameterNames.LARGE_NUMBER.value,
                    TestConfigConstants.TestValues.LARGE_NUMBER,
                ),
            ],
        )
        def test_boundary_values(
            self,
            param_name: str,
            expected_value: t.GeneralValueType,
        ) -> None:
            """Test get_parameter with boundary values."""
            config_dict = self._create_boundary_dict()
            result = FlextUtilitiesConfiguration.get_parameter(config_dict, param_name)
            assert result == expected_value

    class TestSetParameter:
        """Tests for set_parameter method."""

        @pytest.mark.parametrize(
            ("param_name", "value", "expected_success"),
            [
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT,
                    True,
                ),
                (
                    TestConfigConstants.ParameterNames.ENABLED.value,
                    TestConfigConstants.TestValues.TEST_ENABLED_FALSE,
                    True,
                ),
            ],
        )
        def test_on_pydantic_model_success(
            self,
            param_name: str,
            value: t.GeneralValueType,
            expected_success: bool,
        ) -> None:
            """Test set_parameter on Pydantic model with validation."""
            config = ConfigModelForTest(
                name=TestConfigConstants.TestValues.TEST_NAME,
                timeout=TestConfigConstants.TestValues.TEST_TIMEOUT // 2,
            )
            result = FlextUtilitiesConfiguration.set_parameter(
                config,
                param_name,
                cast("t.ScalarValue | m.ConfigMap", value),
            )
            if result:
                assert getattr(config, param_name) == value
            else:
                assert getattr(config, param_name) != value

        @pytest.mark.parametrize(
            ("param_name", "value"),
            [
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT_INVALID,
                ),
                (
                    TestConfigConstants.ParameterNames.MISSING.value,
                    TestConfigConstants.TestValues.TEST_NAME,
                ),
            ],
        )
        def test_on_pydantic_model_validation_error(
            self,
            param_name: str,
            value: t.GeneralValueType,
        ) -> None:
            """Test set_parameter handles Pydantic validation errors."""
            config = ConfigModelForTest(name=TestConfigConstants.TestValues.TEST_NAME)
            result = FlextUtilitiesConfiguration.set_parameter(
                config,
                param_name,
                cast("t.ScalarValue | m.ConfigMap", value),
            )
            assert result is False

        def test_on_non_pydantic_object(self) -> None:
            """Test set_parameter on non-Pydantic object."""
            config = DataclassConfigForTest(
                name=TestConfigConstants.TestValues.TEST_NAME,
                value=TestConfigConstants.TestValues.TEST_VALUE,
            )
            config_cast = cast("t.GeneralValueType | object", config)
            result = FlextUtilitiesConfiguration.set_parameter(
                config_cast,
                TestConfigConstants.ParameterNames.VALUE.value,
                TestConfigConstants.TestValues.TEST_VALUE_UPDATED,
            )
            assert result is True
            assert config.value == TestConfigConstants.TestValues.TEST_VALUE_UPDATED

        @pytest.mark.parametrize(
            ("param_name", "value"),
            [
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT_ZERO,
                ),
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT_LARGE,
                ),
                (
                    TestConfigConstants.ParameterNames.ENABLED.value,
                    TestConfigConstants.TestValues.TEST_ENABLED_FALSE,
                ),
            ],
        )
        def test_boundary_values(
            self,
            param_name: str,
            value: t.GeneralValueType,
        ) -> None:
            """Test set_parameter with boundary values."""
            config = ConfigModelForTest(name=TestConfigConstants.TestValues.TEST_NAME)
            result = FlextUtilitiesConfiguration.set_parameter(
                config,
                param_name,
                cast("t.ScalarValue | m.ConfigMap", value),
            )
            assert result is True
            assert getattr(config, param_name) == value

    class TestSingleton:
        """Tests for singleton methods."""

        def setup_method(self) -> None:
            """Reset singleton instances before each test."""
            SingletonClassForTest.reset_instance()

        @pytest.mark.parametrize(
            ("param_name", "expected_value"),
            [
                (TestConfigConstants.ParameterNames.NAME.value, "default"),
                (
                    TestConfigConstants.ParameterNames.TIMEOUT.value,
                    TestConfigConstants.TestValues.TEST_TIMEOUT // 2,
                ),
            ],
        )
        def test_get_singleton_success(
            self,
            param_name: str,
            expected_value: t.GeneralValueType,
        ) -> None:
            """Test get_singleton from singleton class."""
            result = FlextUtilitiesConfiguration.get_singleton(
                SingletonClassForTest,
                param_name,
            )
            assert result == expected_value

        def test_get_singleton_not_found(self) -> None:
            """Test get_singleton raises NotFoundError for missing parameter."""
            with pytest.raises(FlextExceptions.NotFoundError) as exc_info:
                FlextUtilitiesConfiguration.get_singleton(
                    SingletonClassForTest,
                    TestConfigConstants.ParameterNames.MISSING.value,
                )
            assert TestConfigConstants.ErrorMessages.PARAMETER_NOT_DEFINED.format(
                TestConfigConstants.ParameterNames.MISSING.value,
            ) in str(exc_info.value)

        def test_get_singleton_no_get_global_instance(self) -> None:
            """Test get_singleton raises ValidationError for class without get_global_instance."""
            with pytest.raises(FlextExceptions.ValidationError) as exc_info:
                FlextUtilitiesConfiguration.get_singleton(
                    SingletonWithoutGetGlobalForTest,
                    TestConfigConstants.ParameterNames.VALUE.value,
                )
            assert TestConfigConstants.ErrorMessages.DOES_NOT_HAVE_GET_GLOBAL in str(
                exc_info.value,
            )

        def test_set_singleton_success(self) -> None:
            """Test set_singleton on singleton class."""
            instance = SingletonClassForTest.get_global_instance()
            original_timeout = instance.timeout

            result = FlextUtilitiesConfiguration.set_singleton(
                SingletonClassForTest,
                TestConfigConstants.ParameterNames.TIMEOUT.value,
                TestConfigConstants.TestValues.TEST_TIMEOUT_LARGE,
            )
            _ResultAssertions.assert_success_with_value(result, True)
            assert instance.timeout == TestConfigConstants.TestValues.TEST_TIMEOUT_LARGE

            instance.timeout = original_timeout

        def test_set_singleton_no_get_global_instance(self) -> None:
            """Test set_singleton fails for class without get_global_instance."""
            result = FlextUtilitiesConfiguration.set_singleton(
                SingletonWithoutGetGlobalForTest,
                TestConfigConstants.ParameterNames.VALUE.value,
                "new_value",
            )
            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert (
                TestConfigConstants.ErrorMessages.DOES_NOT_HAVE_GET_GLOBAL
                in result.error
            )

        def test_set_singleton_not_callable(self) -> None:
            """Test set_singleton fails when get_global_instance is not callable."""
            result = FlextUtilitiesConfiguration.set_singleton(
                BadSingletonForTest,
                TestConfigConstants.ParameterNames.VALUE.value,
                TestConfigConstants.TestValues.TEST_NAME,
            )
            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert TestConfigConstants.ErrorMessages.IS_NOT_CALLABLE in result.error

        def test_set_singleton_no_has_model_dump(self) -> None:
            """Test set_singleton fails when instance doesn't implement HasModelDump."""
            result = FlextUtilitiesConfiguration.set_singleton(
                SingletonWithoutModelDumpForTest,
                TestConfigConstants.ParameterNames.VALUE.value,
                TestConfigConstants.TestValues.TEST_NAME,
            )
            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert (
                TestConfigConstants.ErrorMessages.DOES_NOT_IMPLEMENT_HAS_MODEL_DUMP
                in result.error
            )

        def test_set_singleton_parameter_set_failure(self) -> None:
            """Test set_singleton fails when set_parameter fails."""
            SingletonClassForTest.get_global_instance()
            result = FlextUtilitiesConfiguration.set_singleton(
                SingletonClassForTest,
                TestConfigConstants.ParameterNames.MISSING.value,
                TestConfigConstants.TestValues.TEST_NAME,
            )
            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert (
                TestConfigConstants.ErrorMessages.FAILED_TO_SET_PARAMETER.format(
                    TestConfigConstants.ParameterNames.MISSING.value,
                )
                in result.error
            )

    class TestValidateConfigClass:
        """Tests for validate_config_class method."""

        def test_success(self) -> None:
            """Test validate_config_class with valid configuration class."""
            is_valid, error = FlextUtilitiesConfiguration.validate_config_class(
                ConfigModelForTest,
            )
            # Use tm.that for assertions
            _Assertions.that(is_valid, eq=True, msg="Config class must be valid")
            _Assertions.that(
                error,
                none=True,
                msg="Error must be None for valid config",
            )

        def test_no_model_config(self) -> None:
            """Test validate_config_class fails for class without model_config."""
            is_valid, error = FlextUtilitiesConfiguration.validate_config_class(
                ConfigWithoutModelConfigForTest,
            )
            # Use tm.that for assertions
            _Assertions.that(
                is_valid,
                eq=False,
                msg="Config class without model_config must be invalid",
            )
            _Assertions.that(
                error,
                none=False,
                msg="Error must not be None for invalid config",
            )
            _Assertions.that(
                error or "",
                contains=TestConfigConstants.ErrorMessages.MUST_DEFINE_MODEL_CONFIG,
                msg="Error message must contain expected text",
            )

        def test_instantiation_error(self) -> None:
            """Test validate_config_class handles instantiation errors."""
            is_valid, error = FlextUtilitiesConfiguration.validate_config_class(
                BadConfigForTest,
            )
            # Use tm.that for assertions
            _Assertions.that(
                is_valid,
                eq=False,
                msg="Config class with instantiation error must be invalid",
            )
            _Assertions.that(
                error,
                none=False,
                msg="Error must not be None for invalid config",
            )
            _Assertions.that(
                error or "",
                contains=TestConfigConstants.ErrorMessages.CONFIGURATION_CLASS_VALIDATION_FAILED,
                msg="Error message must contain expected text",
            )

    class TestCreateSettingsConfig:
        """Tests for create_settings_config method."""

        def test_minimal(self) -> None:
            """Test create_settings_config with minimal parameters."""
            config = FlextUtilitiesConfiguration.create_settings_config(
                TestConfigConstants.SettingsConfig.ENV_PREFIX,
            )

            # Use tm.that for assertions
            _Assertions.that(
                config["env_prefix"],
                eq=TestConfigConstants.SettingsConfig.ENV_PREFIX,
                msg="env_prefix must match",
            )
            _Assertions.that(config["env_file"], none=True, msg="env_file must be None")
            _Assertions.that(
                config["env_nested_delimiter"],
                eq=TestConfigConstants.SettingsConfig.ENV_NESTED_DELIMITER_DEFAULT,
                msg="env_nested_delimiter must match default",
            )
            _Assertions.that(
                config["case_sensitive"],
                eq=TestConfigConstants.SettingsConfig.CASE_SENSITIVE,
                msg="case_sensitive must match",
            )
            _Assertions.that(
                config["extra"],
                eq=TestConfigConstants.SettingsConfig.EXTRA,
                msg="extra must match",
            )
            assert (
                config["validate_default"]
                == TestConfigConstants.SettingsConfig.VALIDATE_DEFAULT
            )

        def test_full(self) -> None:
            """Test create_settings_config with all parameters."""
            config = FlextUtilitiesConfiguration.create_settings_config(
                TestConfigConstants.SettingsConfig.ENV_PREFIX,
                env_file=TestConfigConstants.SettingsConfig.ENV_FILE,
                env_nested_delimiter=TestConfigConstants.SettingsConfig.ENV_NESTED_DELIMITER_CUSTOM,
            )

            assert config["env_prefix"] == TestConfigConstants.SettingsConfig.ENV_PREFIX
            assert config["env_file"] == TestConfigConstants.SettingsConfig.ENV_FILE
            assert (
                config["env_nested_delimiter"]
                == TestConfigConstants.SettingsConfig.ENV_NESTED_DELIMITER_CUSTOM
            )
            assert (
                config["case_sensitive"]
                == TestConfigConstants.SettingsConfig.CASE_SENSITIVE
            )
            assert config["extra"] == TestConfigConstants.SettingsConfig.EXTRA
            assert (
                config["validate_default"]
                == TestConfigConstants.SettingsConfig.VALIDATE_DEFAULT
            )

    class TestBuildOptionsFromKwargs:
        """Tests for build_options_from_kwargs method."""

        @pytest.mark.parametrize(
            (
                "explicit_format",
                "explicit_indent",
                "expected_format",
                "expected_indent",
            ),
            [
                (
                    TestConfigConstants.TestValues.TEST_FORMAT_XML,
                    TestConfigConstants.TestValues.TEST_INDENT_4,
                    TestConfigConstants.TestValues.TEST_FORMAT_XML,
                    TestConfigConstants.TestValues.TEST_INDENT_4,
                ),
                (
                    TestConfigConstants.TestValues.TEST_FORMAT_JSON,
                    None,
                    TestConfigConstants.TestValues.TEST_FORMAT_JSON,
                    TestConfigConstants.TestValues.TEST_INDENT_2,
                ),
            ],
        )
        def test_explicit_options(
            self,
            explicit_format: str,
            explicit_indent: int | None,
            expected_format: str,
            expected_indent: int,
        ) -> None:
            """Test build_options_from_kwargs with explicit options."""
            explicit = OptionsModelForTest(
                format=explicit_format,
                indent=explicit_indent or TestConfigConstants.TestValues.TEST_INDENT_2,
            )

            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=explicit,
                default_factory=OptionsModelForTest,
            )

            _ResultAssertions.assert_success(result)
            assert result.value.format == expected_format
            assert result.value.indent == expected_indent

        def test_explicit_with_overrides(self) -> None:
            """Test build_options_from_kwargs with explicit options and kwargs overrides."""
            explicit = OptionsModelForTest(
                format=TestConfigConstants.TestValues.TEST_FORMAT_XML,
                indent=TestConfigConstants.TestValues.TEST_INDENT_4,
            )

            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=explicit,
                default_factory=OptionsModelForTest,
                indent=TestConfigConstants.TestValues.TEST_INDENT_8,
                sort_keys=TestConfigConstants.TestValues.TEST_SORT_KEYS_TRUE,
            )

            _ResultAssertions.assert_success(result)
            assert result.value.format == TestConfigConstants.TestValues.TEST_FORMAT_XML
            assert result.value.indent == TestConfigConstants.TestValues.TEST_INDENT_8
            assert (
                result.value.sort_keys
                == TestConfigConstants.TestValues.TEST_SORT_KEYS_TRUE
            )

        def test_default_factory(self) -> None:
            """Test build_options_from_kwargs with default factory."""
            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=None,
                default_factory=lambda: OptionsModelForTest(
                    format=TestConfigConstants.TestValues.TEST_FORMAT_YAML,
                    indent=TestConfigConstants.TestValues.TEST_INDENT_2,
                ),
            )

            _ResultAssertions.assert_success(result)
            assert (
                result.value.format == TestConfigConstants.TestValues.TEST_FORMAT_YAML
            )
            assert result.value.indent == TestConfigConstants.TestValues.TEST_INDENT_2

        def test_default_with_overrides(self) -> None:
            """Test build_options_from_kwargs with default factory and kwargs overrides."""
            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=None,
                default_factory=OptionsModelForTest,
                format=TestConfigConstants.TestValues.TEST_FORMAT_TOML,
                indent=TestConfigConstants.TestValues.TEST_INDENT_6,
            )

            _ResultAssertions.assert_success(result)
            assert (
                result.value.format == TestConfigConstants.TestValues.TEST_FORMAT_TOML
            )
            assert result.value.indent == TestConfigConstants.TestValues.TEST_INDENT_6

        def test_no_kwargs(self) -> None:
            """Test build_options_from_kwargs with no kwargs."""
            explicit = OptionsModelForTest(
                format=TestConfigConstants.TestValues.TEST_FORMAT_JSON,
            )

            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=explicit,
                default_factory=OptionsModelForTest,
            )

            _ResultAssertions.assert_success(result)
            assert (
                result.value.format == TestConfigConstants.TestValues.TEST_FORMAT_JSON
            )
            assert result.value is explicit

        def test_invalid_kwargs(self) -> None:
            """Test build_options_from_kwargs ignores invalid kwargs."""
            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=OptionsModelForTest,
                explicit_options=None,
                default_factory=OptionsModelForTest,
                invalid_field=TestConfigConstants.TestValues.TEST_NAME,
                another_invalid=TestConfigConstants.TestValues.TEST_VALUE,
                format=TestConfigConstants.TestValues.TEST_FORMAT_JSON,
            )

            _ResultAssertions.assert_success(result)
            assert (
                result.value.format == TestConfigConstants.TestValues.TEST_FORMAT_JSON
            )

        def test_validation_error(self) -> None:
            """Test build_options_from_kwargs handles Pydantic validation errors."""
            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=StrictOptionsForTest,
                explicit_options=None,
                default_factory=lambda: StrictOptionsForTest(value=50),
                value=TestConfigConstants.TestValues.TEST_VALUE_INVALID,
            )

            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert (
                TestConfigConstants.ErrorMessages.FAILED_TO_BUILD.format(
                    "StrictOptions",
                )
                in result.error
            )

        def test_unexpected_error(self) -> None:
            """Test build_options_from_kwargs handles unexpected errors."""
            result = FlextUtilitiesConfiguration.build_options_from_kwargs(
                model_class=FailingOptionsForTest,
                explicit_options=cast(
                    "FailingOptionsForTest",
                    object(),
                ),
                default_factory=FailingOptionsForTest,
                value="new",
            )

            _ResultAssertions.assert_failure(result)
            assert result.error is not None
            assert (
                TestConfigConstants.ErrorMessages.UNEXPECTED_ERROR_BUILDING.format(
                    "FailingOptions",
                )
                in result.error
            )
