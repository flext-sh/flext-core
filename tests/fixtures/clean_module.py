"""Clean fixture — every class follows AGENTS.md rules.

Importing this module MUST emit zero enforcement warnings.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

import re
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from typing import Annotated, ClassVar, Final

from typing_extensions import TypeForm

from tests import m, p, r, t, u


class TestsFlextCleanModels:
    """Namespace holder with proper prefix and well-formed inner models."""

    class Core:
        """Core test-model namespace."""

        class Tests:
            """Clean model declarations used by enforcement probes."""

            class GoodEntity(m.ArbitraryTypesModel):
                """Well-formed entity."""

                model_config: ClassVar[m.ConfigDict] = m.ConfigDict(extra="forbid")

                name: Annotated[str, m.Field(description="Entity display name.")] = ""
                tags: Annotated[
                    Sequence[str],
                    m.Field(description="Tag collection."),
                ] = m.Field(default_factory=u.empty(TypeForm(Sequence[str])))
                metadata: Annotated[
                    Mapping[str, str],
                    m.Field(description="Attribute map."),
                ] = m.Field(default_factory=u.empty(TypeForm(Mapping[str, str])))
                aliased_value: Annotated[
                    t.Core.Tests.CentralizedUnion,
                    m.Field(description="Value declared through a t.* alias."),
                ] = None

            class GoodFrozenValue(m.FrozenValueModel):
                """Frozen value object."""

                model_config: ClassVar[m.ConfigDict] = m.ConfigDict(
                    frozen=True,
                    extra="forbid",
                )

                id: Annotated[str, m.Field(description="Opaque value identifier.")]


class TestsFlextCleanConstants:
    """Constants facade — UPPER_CASE names, frozen values."""

    class Core:
        """Core constants namespace."""

        class Tests:
            """Clean immutable constants used by enforcement probes."""

            MAX_RETRIES: Final[int] = 3
            ALLOWED_TAGS: Final[frozenset[str]] = frozenset({"a", "b"})
            BANNER: Final[str] = "clean"


class TestsFlextCleanProtocols:
    """Protocols facade with @runtime_checkable inner protocols."""

    class Core:
        """Core protocols namespace."""

        class Tests(p.Tests):
            """Clean protocol declarations used by enforcement probes."""


class TestsFlextCleanServiceBase:
    """Clean facade stub with no enforcement-triggering MRO requirements."""

    @staticmethod
    def execute() -> p.Result[bool]:
        """Return a stable success result for enforcement import tests.

        Returns:
            A stable success result for enforcement import tests.

        """
        return r[bool].ok(value=True)
