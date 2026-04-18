"""Clean fixture — every class follows AGENTS.md rules.

Importing this module MUST emit zero enforcement warnings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, ClassVar, Final, Protocol, runtime_checkable

from flext_core import FlextModels, m, u


class TestsFlextCoreCleanModels:
    """Namespace holder with proper prefix and well-formed inner models."""

    class Core:
        class Tests:
            class GoodEntity(FlextModels.ArbitraryTypesModel):
                """Well-formed entity."""

                name: Annotated[
                    str,
                    u.Field(description="Entity display name."),
                ] = ""
                tags: Annotated[
                    Sequence[str],
                    u.Field(default_factory=tuple, description="Tag collection."),
                ]
                metadata: Annotated[
                    Mapping[str, str],
                    u.Field(default_factory=dict, description="Attribute map."),
                ]

            class GoodFrozenValue(FlextModels.FrozenValueModel):
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
