"""Clean fixture — every class follows AGENTS.md rules.

Importing this module MUST emit zero enforcement warnings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Annotated, ClassVar, Final, Protocol, override, runtime_checkable

from tests import m, p, r, t, u
from tests.unit.base import TestsFlextCoreServiceBase


class TestsFlextCoreCleanModels:
    """Namespace holder with proper prefix and well-formed inner models."""

    class Core:
        class Tests:
            class GoodEntity(m.ArbitraryTypesModel):
                """Well-formed entity."""

                model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")

                name: Annotated[
                    str,
                    u.Field(description="Entity display name."),
                ] = ""
                tags: Annotated[
                    t.StrSequence,
                    u.Field(default_factory=tuple, description="Tag collection."),
                ]
                metadata: Annotated[
                    t.StrMapping,
                    u.Field(
                        default_factory=lambda: MappingProxyType({}),
                        description="Attribute map.",
                    ),
                ]

            class GoodFrozenValue(m.FrozenValueModel):
                """Frozen value object."""

                model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
                    frozen=True,
                    extra="forbid",
                )

                id: Annotated[
                    str,
                    u.Field(description="Opaque value identifier."),
                ]


class TestsFlextCoreCleanConstants:
    """Constants facade — UPPER_CASE names, frozen values."""

    class Core:
        class Tests:
            MAX_RETRIES: Final[int] = 3
            ALLOWED_TAGS: Final[frozenset[str]] = frozenset({"a", "b"})
            BANNER: Final[str] = "clean"


class TestsFlextCoreCleanProtocols:
    """Protocols facade with @runtime_checkable inner protocols."""

    class Core:
        class Tests:
            @runtime_checkable
            class GoodProtocol(Protocol):
                """Runtime-checkable protocol."""

                def run(self) -> None: ...


class TestsFlextCoreCleanServiceBase(TestsFlextCoreServiceBase[bool]):
    """Service-base facade using the canonical alias-base pattern."""

    @override
    def execute(self) -> p.Result[bool]:
        """Return a stable success result for enforcement import tests."""
        return r[bool].ok(True)
