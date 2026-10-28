"""Real module-bound aliases for runtime and static-only inspection contracts.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import re
from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeAliasType

if TYPE_CHECKING:
    from pathlib import PurePath as DeferredPath


class TestsFlextTypeAliasDeclarations:
    """Importable declarations keep runtime identities tied to their source."""

    class LegalTypes:
        """Aliases with either runtime values or a declared reverse import."""

        type Deferred = DeferredPath | str
        type Resolved = int | str
        type Nested = tuple[Deferred, ...]
        type Factory = Callable[[], Deferred]

    class OtherScope:
        """A homonymous alias keeps an independent native declaration identity."""

        type Deferred = int

    class NestedScopes:
        """Nested homonymous aliases belong to different lexical binding indexes."""

        type Deferred = DeferredPath | str

        class Child:
            """The child does not rebind the enclosing declaration."""

            type Deferred = DeferredPath | bytes

    class InvalidTypes:
        """A missing name without a guarded import retains its NameError."""

        if TYPE_CHECKING:
            type Absent = str

        type Invalid = Absent | str

    class MissingAttribute:
        """A static-only declaration is absent at the runtime attribute boundary."""

        if TYPE_CHECKING:
            type Value = str

    class InvalidAttributeTypes:
        """AttributeError remains a causal failure of the alias evaluator."""

        type Invalid = TestsFlextTypeAliasDeclarations.MissingAttribute.Value

    @staticmethod
    def unbound() -> TypeAliasType:
        """Build a local alias that has no module-bound declaration proving deferral.

        Returns:
            The resulting ``TypeAliasType``.

        """

        class Host:
            type Local = DeferredPath | str

        return Host.Local
