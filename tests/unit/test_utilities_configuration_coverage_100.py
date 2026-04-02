from __future__ import annotations

from enum import StrEnum, unique
from typing import Annotated, ClassVar

import pytest
from pydantic import BaseModel, Field

from flext_tests import tm
from tests import m, p, t

pytestmark = [pytest.mark.unit, pytest.mark.coverage]


class TestFlextUtilitiesConfiguration:
    class Assertions:
        """Helper for value assertions."""

        @staticmethod
        def that(
            value: t.Tests.Testobject,
            *,
            eq: t.Tests.Testobject = None,
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
            result: p.Result[T],
            expected: T,
        ) -> None:
            tm.that(getattr(result, "is_success"), eq=True)
            tm.that(getattr(result, "value"), eq=expected)

        @staticmethod
        def assert_success[T](result: p.Result[T]) -> None:
            tm.that(getattr(result, "is_success"), eq=True)

        @staticmethod
        def assert_failure[T](result: p.Result[T]) -> None:
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
            int,
            Field(default=42, description="Config t.NormalizedValue value"),
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
            msg = "Must use unified test helpers per Rule 3.6"
            raise NotImplementedError(msg)

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

    def _create_test_dict(self) -> t.ContainerMapping:
        return {
            self.ParameterNames.NAME.value: "test",
            self.ParameterNames.TIMEOUT.value: 60,
            self.ParameterNames.ENABLED.value: True,
        }
