from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from enum import StrEnum, unique
from typing import Annotated, Any, ClassVar, cast

import pytest
from flext_tests import t, tm
from pydantic import BaseModel, Field

from flext_core import FlextExceptions, FlextRuntime
from tests import c, m, p, u

from ._models import TestUnitModels

pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesConfiguration:
    class Assertions:
        """Helper for value assertions."""

        @staticmethod
        def that(
            value: t.NormalizedValue,
            *,
            eq: t.NormalizedValue = None,
            none: bool | None = None,
            contains: str | None = None,
        ) -> None:
            if eq is not None:
                tm.that(value, eq=eq)
            if none is True:
                tm.that(value, eq=None)
            if none is False:
                tm.that(value, none=False)
            if contains is not None:
                assert isinstance(value, str)
                tm.that(value, has=contains)

    class ResultAssertions:
        """Helper for result assertions."""

        @staticmethod
        def assert_success_with_value[T: t.NormalizedValue](
            result: FlextRuntime.RuntimeResult[T],
            expected: T,
        ) -> None:
            tm.that(getattr(result, "is_success"), eq=True)
            tm.that(getattr(result, "value"), eq=expected)

        @staticmethod
        def assert_success[T](result: FlextRuntime.RuntimeResult[T]) -> None:
            tm.that(getattr(result, "is_success"), eq=True)

        @staticmethod
        def assert_failure[T](result: FlextRuntime.RuntimeResult[T]) -> None:
            tm.that(getattr(result, "is_failure"), eq=True)

    class OptionsModelForTest(m.Value):
        """Test options model with format, indent, and sort_keys."""

        format: str = "json"
        indent: int = 2
        sort_keys: bool = False

    class StrictOptionsForTest(m.Value):
        """Test options with strict value range."""

        value: Annotated[int, Field(ge=0, le=100)]

    class DataclassConfigForTest(BaseModel):
        """Test config with name and value."""

        name: Annotated[str, Field(description="Config t.NormalizedValue name")]
        value: Annotated[
            int, Field(default=42, description="Config t.NormalizedValue value")
        ] = 42

    class SingletonWithoutGetGlobalForTest:
        """Singleton without get_global method."""

        def __init__(self) -> None:
            self.value = "test"

    class BadSingletonForTest:
        """Singleton with non-callable get_global."""

        get_global = "not callable"

    class SingletonWithoutModelDumpForTest:
        """Singleton without model_dump method."""

        _instance: ClassVar[
            TestFlextUtilitiesConfiguration.SingletonWithoutModelDumpForTest | None
        ] = None

        @classmethod
        def get_global(
            cls,
        ) -> TestFlextUtilitiesConfiguration.SingletonWithoutModelDumpForTest:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    class ConfigWithoutModelConfigForTest:
        """Config without model_config attribute."""

        def __init__(self) -> None:
            pass

    class FailingOptionsForTest(m.Value):
        """Test options that fail validation."""

        value: str = "test"

    @unique
    class ParameterNames(StrEnum):
        """Parameter names enumeration."""

        NAME = "name"
        TIMEOUT = "timeout"
        ENABLED = "enabled"
        VALUE = "value"
        MISSING = "missing"
        FORMAT = "format"
        INDENT = "indent"
        SORT_KEYS = "sort_keys"

    def _create_test_dict(self) -> Mapping[str, t.NormalizedValue]:
        return {
            self.ParameterNames.NAME.value: "test",
            self.ParameterNames.TIMEOUT.value: 60,
            self.ParameterNames.ENABLED.value: True,
        }

    def test_get_parameter_from_dict(self) -> None:
        config_dict = self._create_test_dict()
        self.Assertions.that(
            u.get_parameter(config_dict, self.ParameterNames.NAME.value),
            eq="test",
        )
        self.Assertions.that(
            u.get_parameter(config_dict, self.ParameterNames.TIMEOUT.value),
            eq=60,
        )

    def test_get_parameter_from_dict_not_found(self) -> None:
        with pytest.raises(FlextExceptions.NotFoundError):
            u.get_parameter(self._create_test_dict(), self.ParameterNames.MISSING.value)

    def test_get_parameter_from_pydantic_model(self) -> None:
        config = TestUnitModels.ConfigModelForTest(
            name="test", timeout=60, enabled=False
        )
        result = u.get_parameter(config, self.ParameterNames.TIMEOUT.value)
        self.Assertions.that(result, eq=60)

    def test_get_parameter_from_attribute_access(self) -> None:
        config = self.DataclassConfigForTest(name="test", value=42)
        config_cast: p.HasModelDump | MutableMapping[str, t.NormalizedValue] = config
        result = u.get_parameter(config_cast, self.ParameterNames.VALUE.value)
        tm.that(result, eq=42)

    def test_get_parameter_from_duck_model_dump_path(self) -> None:
        class _DumpOnly:
            __slots__ = ()

            def model_dump(self) -> Mapping[str, t.NormalizedValue]:
                return {"timeout": 77}

        result = u.get_parameter(cast("p.HasModelDump", _DumpOnly()), "timeout")
        tm.that(result, eq=77)

    @pytest.mark.parametrize(
        ("param_name", "value"),
        [("timeout", 60), ("enabled", False)],
    )
    def test_set_parameter_on_pydantic_model(
        self,
        param_name: str,
        value: t.NormalizedValue,
    ) -> None:
        config = TestUnitModels.ConfigModelForTest(name="test", timeout=30)
        result = u.set_parameter(config, param_name, value)
        tm.that(result, eq=True)
        tm.that(getattr(config, param_name), eq=value)

    def test_set_parameter_validation_error(self) -> None:
        config = TestUnitModels.ConfigModelForTest(name="test")
        result = u.set_parameter(config, self.ParameterNames.TIMEOUT.value, -1)
        tm.that(result, eq=False)

    def test_set_parameter_non_pydantic_object(self) -> None:
        config = self.DataclassConfigForTest(name="test", value=42)
        result = u.set_parameter(config, self.ParameterNames.VALUE.value, 100)
        tm.that(result, eq=True)
        tm.that(config.value, eq=100)

    def test_get_singleton_success(self) -> None:
        TestUnitModels.SingletonClassForTest.reset_instance()
        result = u.get_singleton(TestUnitModels.SingletonClassForTest, "name")
        tm.that(result, eq="default")

    def test_get_singleton_no_get_global(self) -> None:
        with pytest.raises(FlextExceptions.ValidationError):
            u.get_singleton(self.SingletonWithoutGetGlobalForTest, "value")

    def test_set_singleton_success(self) -> None:
        instance = TestUnitModels.SingletonClassForTest.get_global()
        result = u.set_singleton(
            TestUnitModels.SingletonClassForTest,
            "timeout",
            1000,
        )
        self.ResultAssertions.assert_success_with_value(result, True)
        tm.that(instance.timeout, eq=1000)

    def test_set_singleton_failures(self) -> None:
        result_no_get_global = u.set_singleton(
            self.SingletonWithoutGetGlobalForTest,
            "value",
            "new_value",
        )
        self.ResultAssertions.assert_failure(result_no_get_global)

        result_not_callable = u.set_singleton(
            self.BadSingletonForTest,
            "value",
            "test",
        )
        self.ResultAssertions.assert_failure(result_not_callable)

        result_no_model_dump = u.set_singleton(
            self.SingletonWithoutModelDumpForTest,
            "value",
            "test",
        )
        self.ResultAssertions.assert_failure(result_no_model_dump)

    def test_validate_config_class(self) -> None:
        valid_result = u.validate_config_class(TestUnitModels.ConfigModelForTest)
        self.ResultAssertions.assert_success(valid_result)
        self.Assertions.that(valid_result.value, eq=True)

        invalid_result = u.validate_config_class(self.ConfigWithoutModelConfigForTest)
        self.ResultAssertions.assert_failure(invalid_result)

    def test_create_settings_config(self) -> None:
        config = u.create_settings_config("MYAPP_")
        tm.that(config["env_prefix"], eq="MYAPP_")
        tm.that(config["env_file"], eq=c.ENV_FILE_DEFAULT)
        tm.that(config["env_nested_delimiter"], eq="__")

        custom = u.create_settings_config(
            "MYAPP_",
            env_file=".env.test",
            env_nested_delimiter="::",
        )
        tm.that(custom["env_file"], eq=".env.test")
        tm.that(custom["env_nested_delimiter"], eq="::")

    def test_build_options_from_kwargs_explicit(self) -> None:
        explicit = self.OptionsModelForTest(format="xml", indent=4)
        result = u.build_options_from_kwargs(
            model_class=self.OptionsModelForTest,
            explicit_options=explicit,
            default_factory=self.OptionsModelForTest,
        )
        self.ResultAssertions.assert_success(result)
        tm.that(result.value.format, eq="xml")
        tm.that(result.value.indent, eq=4)

    def test_build_options_from_kwargs_with_overrides(self) -> None:
        result = u.build_options_from_kwargs(
            model_class=self.OptionsModelForTest,
            explicit_options=None,
            default_factory=self.OptionsModelForTest,
            format="toml",
            indent=6,
        )
        self.ResultAssertions.assert_success(result)
        tm.that(result.value.format, eq="toml")
        tm.that(result.value.indent, eq=6)

    def test_build_options_from_kwargs_validation_error(self) -> None:
        result = u.build_options_from_kwargs(
            model_class=self.StrictOptionsForTest,
            explicit_options=None,
            default_factory=lambda: self.StrictOptionsForTest(value=50),
            value=200,
        )
        self.ResultAssertions.assert_failure(result)

    def test_build_options_from_kwargs_unexpected_error(self) -> None:
        result = u.build_options_from_kwargs(
            model_class=self.FailingOptionsForTest,
            explicit_options=cast("Any", "normalized"),
            default_factory=self.FailingOptionsForTest,
            value="new",
        )
        self.ResultAssertions.assert_failure(result)
