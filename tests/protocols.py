"""Protocol definitions for flext-core tests.

Provides TestsFlextProtocols, extending TestsFlextProtocols with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- TestsFlextProtocols (flext_tests) = Generic protocols for all FLEXT projects
- TestsFlextProtocols (tests/) = flext-core-specific protocols extending TestsFlextProtocols

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Mapping,
    Sequence,
)
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import p

from tests import t

if TYPE_CHECKING:
    from tests import TestsFlextModelsMixins, TestsFlextModelsMixins as _Mixins

    AttrObject = _Mixins.AttrObject
    BadMapping = _Mixins.BadMapping


class TestsFlextProtocols(p):
    """Protocol definitions for flext-core tests - extends TestsFlextProtocols.

    Architecture: Extends TestsFlextProtocols with flext-core-specific protocol
    definitions. All generic protocols from TestsFlextProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from TestsFlextProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from TestsFlextProtocols
    """

    class Tests(p.Tests):
        """flext-core test protocols namespace."""

        @runtime_checkable
        class ExtractFieldCallable(Protocol):
            """Protocol for _extract_field_value callable."""

            def __call__(self, item: AttrObject, field_name: str) -> t.JsonValue: ...

        @runtime_checkable
        class TakeCallable(Protocol):
            """Protocol for take callable."""

            def __call__(
                self,
                data_or_items: TestsFlextModelsMixins.MaybeModel
                | TestsFlextModelsMixins.PortModel
                | int,
                key_or_index: int | str,
                *,
                default: str | None = None,
            ) -> t.JsonMapping | t.JsonList | t.JsonValue: ...

        @runtime_checkable
        class BuildApplyConvertCallable(Protocol):
            """Protocol for _op_convert callable."""

            def __call__(
                self,
                current: tuple[str, ...] | str | int,
                operations: Mapping[str, t.MapperInput],
                default_val: t.JsonValue,
                on_error: str,
            ) -> t.JsonValue: ...

        @runtime_checkable
        class ExtractTransformOptionsCallable(Protocol):
            """Protocol for _extract_transform_options callable."""

            def __call__(
                self,
                transform_opts: Mapping[str, t.MapperInput],
            ) -> tuple[
                bool,
                bool,
                bool,
                t.StrMapping | None,
                set[str] | None,
                set[str] | None,
            ]: ...

        @runtime_checkable
        class BuildApplyOpCallable(Protocol):
            """Protocol for op callable (sort/unique/slice/group)."""

            def __call__(
                self,
                current: tuple[str, str]
                | tuple[int, int, int]
                | Sequence[TestsFlextModelsMixins.GroupModel],
                operations: Mapping[str, t.MapperInput],
                default_val: t.JsonValue,
                on_error: str,
            ) -> t.JsonMapping | t.JsonList | t.JsonValue: ...

        @runtime_checkable
        class TransformCallable(Protocol):
            """Protocol for transform callable."""

            def __call__(
                self,
                source: BadMapping,
                **kwargs: t.StrMapping,
            ) -> p.Result[t.JsonMapping]: ...

        @runtime_checkable
        class MapDictKeysCallable(Protocol):
            """Protocol for map_dict_keys callable."""

            def __call__(
                self,
                source: TestsFlextModelsMixins.BadItems,
                key_map: t.StrMapping,
                *,
                keep_unmapped: bool = True,
            ) -> p.Result[t.JsonMapping]: ...


p = TestsFlextProtocols
__all__: list[str] = ["TestsFlextProtocols", "p"]
