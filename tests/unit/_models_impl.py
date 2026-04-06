"""Auto-generated centralized models."""

from __future__ import annotations

from collections.abc import Mapping, MutableSequence, Sequence
from typing import Annotated, ClassVar, Never, override

from pydantic import BaseModel, ConfigDict, Field

from tests import t


class _ValidationLikeError(Exception):
    """Validation-like error for tests."""

    def errors(self) -> Sequence[t.ContainerMapping]:
        return [{"loc": ["value"], "msg": "bad value"}]


type TestCaseMap = Mapping[str, t.Tests.TestobjectSerializable]

type InputPayloadMap = Mapping[str, t.Tests.TestobjectSerializable]

__all__ = [
    "BadConfigForTest",
    "CacheTestModel",
    "ComplexModel",
    "ConfigModelForTest",
    "InputPayloadMap",
    "InvalidModelForTest",
    "NestedModel",
    "SampleModel",
    "SingletonClassForTest",
    "TestCaseMap",
    "_BadCopyModel",
    "_BrokenDumpModel",
    "_Cfg",
    "_DumpErrorModel",
    "_ErrorsModel",
    "_FakeConfig",
    "_FrozenEntity",
    "_GoodModel",
    "_Model",
    "_MsgWithCommandId",
    "_MsgWithMessageId",
    "_Opts",
    "_PlainErrorModel",
    "_SampleEntity",
    "_SvcModel",
    "_TargetModel",
    "_ValidationLikeError",
]


class _MsgWithCommandId(BaseModel):
    command_id: str = "cmd-1"


class _MsgWithMessageId(BaseModel):
    message_id: str = "msg-1"


class SampleModel(BaseModel):
    """Sample model for testing."""

    name: str
    value: int


class _SvcModel(BaseModel):
    value: str


class _BrokenDumpModel(BaseModel):
    value: int = 1

    @override
    def __getattribute__(self, name: str) -> t.Tests.PredicateSpec | None:
        if name == "model_dump":

            def _broken_dump(
                _value: t.Tests.Testobject = None,
            ) -> bool:
                return True

            return _broken_dump
        return super().__getattribute__(name)


class _ErrorsModel(BaseModel):
    value: int

    @classmethod
    @override
    def model_validate(
        cls,
        obj: t.NormalizedValue,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: t.ContainerMapping | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Never:
        _ = strict, extra, from_attributes, context, by_alias, by_name
        _ = obj
        raise _ValidationLikeError


class _PlainErrorModel(BaseModel):
    value: int

    @classmethod
    @override
    def model_validate(
        cls,
        obj: t.NormalizedValue,
        *,
        strict: bool | None = None,
        extra: str | None = None,
        from_attributes: bool | None = None,
        context: t.ContainerMapping | None = None,
        by_alias: bool | None = None,
        by_name: bool | None = None,
    ) -> Never:
        _ = strict, extra, from_attributes, context, by_alias, by_name
        _ = obj
        msg = "plain boom"
        raise RuntimeError(msg)


class _TargetModel(BaseModel):
    value: int


class CacheTestModel(BaseModel):
    """Test model for cache key generation."""

    name: str
    value: int
    tags: MutableSequence[str] = []
    meta: t.MutableStrMapping = {}


class NestedModel(BaseModel):
    """Nested Pydantic model for cache testing."""

    inner: CacheTestModel
    count: int


class ConfigModelForTest(BaseModel):
    """Test configuration model (mutable for set_parameter tests)."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    name: str = "default_config"
    timeout: Annotated[int, Field(default=30, ge=0)] = 30
    enabled: bool = True


class InvalidModelForTest(BaseModel):
    """Model with invalid model_dump."""

    value: str = "test"


class SingletonClassForTest(BaseModel):
    """Test singleton class with Pydantic validation."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        validate_assignment=True,
        extra="forbid",
    )

    _instance: ClassVar[SingletonClassForTest | None] = None

    name: str = "default"
    timeout: int = 30

    @classmethod
    def get_global(cls) -> SingletonClassForTest:
        """Get global singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance for test isolation."""
        cls._instance = None


class BadConfigForTest(BaseModel):
    """Config that fails to instantiate."""

    model_config: ClassVar[ConfigDict] = ConfigDict(validate_assignment=True)

    def __init__(self, **kwargs: t.Scalar) -> None:
        """Raise error on init."""
        super().__init__(**kwargs)
        msg = "Cannot instantiate"
        raise ValueError(msg)


class _DumpErrorModel(BaseModel):
    value: int = 1


class _Opts(BaseModel):
    value: int = 1


class _FakeConfig(BaseModel):
    """Fake config with model_copy support."""

    timeout: int = 10

    @property
    def data(self) -> t.ContainerMapping:
        return {"timeout": self.timeout}


class _Model(BaseModel):
    value: int


class _SampleEntity(BaseModel):
    """Test entity for domain utility tests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=False)

    unique_id: str = "test-123"
    name: str = "test"


class _FrozenEntity(BaseModel):
    """Frozen entity for immutability tests."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    unique_id: str = "frozen-1"


class _GoodModel(BaseModel):
    value: int = 7


class ComplexModel(BaseModel):
    """Complex test model."""

    id: int
    data: t.ContainerMapping
    items: t.StrSequence


class _Cfg(BaseModel):
    x: int = 0
    y: str = "a"


class _BadCopyModel(BaseModel):
    x: int = 1
