"""Core state model helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, ClassVar

from flext_core import m
from tests.constants import c

if TYPE_CHECKING:
    from collections.abc import MutableSequence

    from tests.typings import p, t


class TestsFlextModelsCoreStateMixin:
    """Core state model helpers."""

    class CacheTestModel(m.BaseModel):
        """Test model for cache key generation."""

        name: str
        value: int
        tags: MutableSequence[str] = []
        meta: t.MutableStrMapping = {}

    class NestedModel(m.BaseModel):
        """Nested Pydantic model for cache testing."""

        inner: TestsFlextModelsCoreStateMixin.CacheTestModel
        count: int

    class SettingsModelForTest(m.BaseModel):
        """Test configuration model (mutable for set_parameter tests)."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        name: str = "default_settings"
        timeout: Annotated[int, m.Field(ge=0)] = 30
        enabled: bool = True

    class InvalidModelForTest(m.BaseModel):
        """Model with invalid model_dump."""

        value: str = "test"

    class SingletonClassForTest(m.BaseModel):
        """Test singleton class with Pydantic validation."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
            extra="forbid",
        )

        _instance: ClassVar[
            TestsFlextModelsCoreStateMixin.SingletonClassForTest | None
        ] = None

        name: str = "default"
        timeout: int = 30

        @classmethod
        def fetch_global(
            cls,
        ) -> TestsFlextModelsCoreStateMixin.SingletonClassForTest:
            """Get global singleton instance."""
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

        @classmethod
        def reset_instance(cls) -> None:
            """Reset singleton instance for test isolation."""
            cls._instance = None

    class BadSettingsForTest(m.BaseModel):
        """Settings that fails to instantiate."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(
            validate_assignment=True,
        )

        def __init__(self, **kwargs: t.Scalar) -> None:
            """Raise error on init."""
            super().__init__(**kwargs)
            msg = c.Tests.CANNOT_INSTANTIATE
            raise ValueError(msg)

    class _DumpErrorModel(m.BaseModel):
        value: int = 1

    class _Opts(m.BaseModel):
        value: int = 1

    class _FakeSettings(m.BaseModel):
        """Fake settings with model_copy support."""

        timeout: int = 10

        @property
        def data(self) -> t.JsonMapping:
            return {"timeout": self.timeout}

    class _Model(m.BaseModel):
        value: int

    class _SampleEntity(m.BaseModel):
        """Test entity for domain utility tests."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=False)

        unique_id: str = "test-123"
        name: str = "test"

    class _FrozenEntity(m.BaseModel):
        """Frozen entity for immutability tests."""

        model_config: ClassVar[t.ConfigDict] = m.ConfigDict(frozen=True)

        unique_id: str = "frozen-1"

    class _GoodModel(m.BaseModel):
        value: int = 7

    class ComplexModel(m.BaseModel):
        """Complex test model."""

        id: int
        data: t.JsonMapping
        items: t.StrSequence

    class _Cfg(m.BaseModel):
        x: int = 0
        y: str = "a"

    class _BadCopyModel(m.BaseModel):
        x: int = 1


__all__: list[str] = ["TestsFlextModelsCoreStateMixin"]
