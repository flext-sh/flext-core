"""Core error model helpers."""

from __future__ import annotations

from collections import UserDict, UserList
from typing import TYPE_CHECKING, Annotated, Never, override

from flext_core import m
from tests.constants import c
from tests.typings import t

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class TestsFlextModelsCoreErrorsMixin:
    """Core error model helpers."""

    class BadDict(UserDict[str, t.Tests.TestobjectSerializable]):
        """Dict that raises on get()."""

        @override
        def __getitem__(self, key: str) -> Never:
            """Raise error on get attempt."""
            _ = key
            msg = c.Tests.BAD_DICT_GET
            raise RuntimeError(msg)

    class BadList(UserList[t.Tests.TestobjectSerializable]):
        """List that raises on iteration."""

        @override
        def __iter__(self) -> Iterator[t.Tests.TestobjectSerializable]:
            """Raise error on iteration."""
            msg = c.Tests.BAD_LIST_ITERATION
            raise RuntimeError(msg)

    class BadModelDump:
        """Object with model_dump that raises."""

        model_dump: Callable[[], t.MappingKV[str, t.Tests.TestobjectSerializable]] = (
            staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("Bad model_dump")))
        )

    class AttrObject(m.BaseModel):
        """Simple model with name/value attributes for mapper tests."""

        name: Annotated[
            str, m.Field(description="Attribute recursive container name")
        ] = "name"
        value: Annotated[
            int, m.Field(description="Attribute recursive container value")
        ] = 1

    class BadMapping(UserDict[str, t.JsonValue]):
        """Mapping that raises on access — used for error-path testing."""

        @override
        def __getitem__(self, key: str) -> t.JsonValue:
            _ = key
            msg = "get exploded"
            raise TypeError(msg)

        @override
        def __iter__(self) -> Iterator[str]:
            msg = "iter exploded"
            raise TypeError(msg)

        @override
        def __len__(self) -> int:
            return 1

    type TestCaseMap = t.MappingKV[str, t.Tests.TestobjectSerializable]
    type InputPayloadMap = t.MappingKV[str, t.Tests.TestobjectSerializable]

    class SampleModel(m.BaseModel):
        """Sample model for testing."""

        name: str
        value: int

    class BrokenDumpModel:
        """Test fake whose ``model_dump`` returns wrong type.

        Triggers ``TypeError`` in mapping-shaped validators (e.g.,
        ``Metadata.attributes``). Intentionally NOT a ``m.BaseModel`` subclass:
        avoids ``__getattribute__`` override (forbidden outside flext-core
        src/) while still presenting the duck-typed ``model_dump`` callable
        Pydantic runtime probes.
        """

        value: int = 1

        @staticmethod
        def model_dump() -> bool:
            return True

    class TargetModel(m.BaseModel):
        value: int


__all__: list[str] = ["TestsFlextModelsCoreErrorsMixin"]
