"""Layer-specific enforcement tests."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod
from typing import Final, Protocol, runtime_checkable

from tests.unit._enforcement_support import messages
from tests.utilities import u

if typing.TYPE_CHECKING:
    from tests.typings import t


class TestsFlextEnforcementLayers:
    def test_mutable_list_detected(self) -> None:
        class _CConstants:
            ITEMS: list[str] = ["a", "b"]

        assert messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_mutable_dict_detected(self) -> None:
        class _CConstants:
            DATA: dict[str, int] = {"x": 1}

        assert messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_frozenset_passes(self) -> None:
        class _CConstants:
            ITEMS: Final[frozenset[str]] = frozenset({"a"})

        assert not messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_tuple_passes(self) -> None:
        class _CConstants:
            ITEMS: Final[t.StrSequence] = ("a", "b")

        assert not messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_lowercase_constant_detected(self) -> None:
        class _CConstants:
            my_value: int = 42

        assert messages(
            u.check(_CConstants, layer="constants"),
            fragment="UPPER_CASE",
        )

    def test_inner_namespace_mutable_detected(self) -> None:
        class _CConstants:
            class Inner:
                BAD: list[str] = ["x"]

        assert messages(
            u.check(_CConstants, layer="constants"),
            fragment="mutable constant",
        )

    def test_non_protocol_inner_detected(self) -> None:
        class _PProtocols:
            class NotAProtocol:
                pass

        assert messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="Protocol",
        )

    def test_abc_passes(self) -> None:
        class _PProtocols:
            class SomeContract(ABC):
                @abstractmethod
                def do(self) -> None: ...

        msgs = messages(u.check(_PProtocols, layer="protocols"), fragment="Protocol")
        assert not any("must be Protocol" in m for m in msgs)

    def test_non_runtime_protocol_detected(self) -> None:
        class _PProtocols:
            class InnerProto(Protocol):
                def do(self) -> None: ...

        assert messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="runtime_checkable",
        )

    def test_runtime_protocol_passes(self) -> None:
        class _PProtocols:
            @runtime_checkable
            class InnerProto(Protocol):
                def do(self) -> None: ...

        assert not messages(
            u.check(_PProtocols, layer="protocols"),
            fragment="runtime_checkable",
        )

    def test_alias_with_any_detected(self) -> None:
        class _TTypes:
            type BadAlias = typing.Any

        assert messages(u.check(_TTypes, layer="types"), fragment="Any in type alias")

    def test_clean_alias_passes(self) -> None:
        class _TTypes:
            type GoodAlias = str

        assert not messages(
            u.check(_TTypes, layer="types"),
            fragment="Any in type alias",
        )

    def test_instance_method_detected(self) -> None:
        class _UUtilities:
            def run(self) -> None: ...

        assert messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )

    def test_static_method_passes(self) -> None:
        class _UUtilities:
            @staticmethod
            def run() -> None: ...

        assert not messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )

    def test_class_method_passes(self) -> None:
        class _UUtilities:
            @classmethod
            def run(cls) -> None: ...

        assert not messages(
            u.check(_UUtilities, layer="utilities"),
            fragment="staticmethod",
        )
