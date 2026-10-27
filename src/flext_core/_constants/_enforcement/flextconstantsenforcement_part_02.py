"""Runtime enforcement constants for FlextConstantsEnforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants._enforcement import FlextConstantsEnforcementEnums

if TYPE_CHECKING:
    from collections.abc import Mapping

    from flext_core import t


class FlextConstantsEnforcementRuntime:
    """Runtime modes, base exemptions, and collection contracts."""

    ENFORCEMENT_MODE: ClassVar[FlextConstantsEnforcementEnums.EnforcementMode] = (
        FlextConstantsEnforcementEnums.EnforcementMode.WARN
    )
    """Controls behavior: strict (TypeError), warn (UserWarning), off."""

    ENFORCEMENT_FORBIDDEN_COLLECTIONS: ClassVar[Mapping[type, str]] = MappingProxyType({
        dict: "Mapping[K, V] or t.JsonMapping",
        list: "Sequence[X] or t.JsonList",
        set: "frozenset[X] or AbstractSet[X]",
    })
    """SSOT: forbidden mutable-collection types mapped to replacement hints.

    Downstream constants (``ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS`` as
    name set, ``ENFORCEMENT_MUTABLE_RUNTIME_TYPES`` as runtime tuple) are
    derived from this single mapping — do not maintain parallel lists.
    """

    ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS: ClassVar[frozenset[str]] = frozenset(
        kind.__name__ for kind in ENFORCEMENT_FORBIDDEN_COLLECTIONS
    )
    """Derived view: collection names used by annotation-origin checks."""

    ENFORCEMENT_MUTABLE_RUNTIME_TYPES: ClassVar[t.VariadicTuple[type]] = tuple(
        ENFORCEMENT_FORBIDDEN_COLLECTIONS,
    )
    """Derived view: concrete types used by ``isinstance`` checks."""


__all__: list[str] = ["FlextConstantsEnforcementRuntime"]
