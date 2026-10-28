"""Protocol definitions for flext-core tests.

Provides TestsFlextProtocols, extending TestsFlextProtocols with flext-core-specific
protocols. All generic test protocols come from flext_tests.

Architecture:
- TestsFlextProtocols (flext_tests) = Generic protocols for all FLEXT projects
- TestsFlextProtocols (tests/) = flext-core-specific protocols
  extending TestsFlextProtocols

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from flext_tests import FlextTestsProtocols

from flext_core import c

if TYPE_CHECKING:
    from tests import c, t
    from tests._models import TestsFlextModelsMixins


class TestsFlextProtocols(FlextTestsProtocols):
    """Protocol definitions for flext-core tests - extends TestsFlextProtocols.

    Architecture: Extends TestsFlextProtocols with flext-core-specific protocol
    definitions. All generic protocols from TestsFlextProtocols are available
    through inheritance.

    Rules:
    - NEVER redeclare protocols from TestsFlextProtocols
    - Only flext-core-specific protocols allowed
    - All generic protocols come from TestsFlextProtocols
    """

    class Tests(FlextTestsProtocols.Tests):
        """flext-core test protocols namespace."""

        class InferredCommand(Protocol):
            """A request route hint does not restrict runtime handler capability."""

            command_type: str = c.DEFAULT_COMMAND_TYPE

        @runtime_checkable
        class DispatchRequest(Protocol):
            """Only the request capability consumed by the dispatch operation."""

            @property
            def command_name(self) -> str:
                """Expose the name consumed by the dispatch operation."""
                ...

        @runtime_checkable
        class PublicPayload(FlextTestsProtocols.Base, Protocol):
            """Data consumed through a protocol and validated by its model owner."""

            raw_name: str
            visits: int

        @runtime_checkable
        class GoodProtocol(FlextTestsProtocols.Base, Protocol):
            """Runtime-checkable port used by the clean enforcement fixture."""

            @staticmethod
            def run() -> None:
                """Execute the clean protocol behavior."""
                ...

        @runtime_checkable
        class Counter(FlextTestsProtocols.Base, Protocol):
            """Dependency port of the service contract tests: a monotonic counter."""

            def next_value(self) -> int:
                """Advance the counter and return its new value."""
                ...

        @runtime_checkable
        class ExtractFieldCallable(Protocol):
            """Protocol for _extract_field_value callable."""

            def __call__(
                self,
                item: TestsFlextModelsMixins.TestsFlextModelsMixins.AttrObject,
                field_name: str,
            ) -> t.JsonValue:
                """Extract one named field value from an attribute object."""
                ...

        @runtime_checkable
        class TakeCallable(Protocol):
            """Protocol for take callable."""

            def __call__(
                self,
                data_or_items: TestsFlextModelsMixins.TestsFlextModelsMixins.MaybeModel
                | TestsFlextModelsMixins.TestsFlextModelsMixins.PortModel
                | int,
                key_or_index: int | str,
                *,
                default: str | None = None,
            ) -> t.JsonMapping | t.JsonList | t.JsonValue:
                """Take one key or index from supported mapper input."""
                ...

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
                Mapping[str, str] | None,
                set[str] | None,
                set[str] | None,
            ]:
                """Extract normalized transform options from mapper input."""
                ...

        @runtime_checkable
        class TransformCallable(Protocol):
            """Protocol for transform callable."""

            def __call__(
                self,
                source: TestsFlextModelsMixins.TestsFlextModelsMixins.BadMapping,
                **kwargs: Mapping[str, str],
            ) -> p.Result[t.JsonMapping]:
                """Transform one mapping source into a result mapping."""
                ...

        @runtime_checkable
        class MapDictKeysCallable(Protocol):
            """Protocol for map_dict_keys callable."""

            def __call__(
                self,
                source: TestsFlextModelsMixins.TestsFlextModelsMixins.BadItems,
                key_map: Mapping[str, str],
                *,
                keep_unmapped: bool = True,
            ) -> p.Result[t.JsonMapping]:
                """Map source dictionary keys under the requested policy."""
                ...


p = TestsFlextProtocols

__all__: list[str] = ["TestsFlextProtocols", "p"]
