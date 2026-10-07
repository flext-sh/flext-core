"""Runtime enforcement constants for FlextConstantsEnforcement.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_01 import (
    FlextConstantsEnforcementEnums,
)
from flext_core._typings.base import FlextTypingBase

if TYPE_CHECKING:
    from collections.abc import Mapping


class FlextConstantsEnforcementRuntime:
    """Runtime modes, base exemptions, and collection contracts."""

    ENFORCEMENT_MODE: ClassVar[FlextConstantsEnforcementEnums.EnforcementMode] = (
        FlextConstantsEnforcementEnums.EnforcementMode.WARN
    )
    """Controls behavior: strict (TypeError), warn (UserWarning), off."""

    BEARTYPE_MODE: ClassVar[FlextConstantsEnforcementEnums.EnforcementMode] = (
        FlextConstantsEnforcementEnums.EnforcementMode.OFF
    )
    """Controls flext_core beartype.claw bootstrap: strict, warn, or off.

    Compile-time constant (ENFORCE-037 forbids ``os.environ`` reads; no code
    reads an env var). ``warn`` surfaces runtime type violations as
    ``UserWarning``; ``strict`` raises ``TypeError``; ``off`` disables the claw
    bootstrap. Change the value here to flip.

    Currently ``off``. The ``pydantic.JsonValue`` forward-ref crash that blocked
    WARN is fixed in ``beartype_typingext_patch`` (bead mro-31mj.2). WARN is
    still blocked by a second upstream beartype defect: decorating a class whose
    nested ``@beartype``-decorated class defines ``__init__`` shadows the outer
    class's *inherited* ``__init__`` with the nested one, breaking the c/m/p/t/u
    nested-class facades (e.g. ``FlextUtilitiesLogging.PerformanceTracker``). Fixing it
    requires changes to beartype's core class decorator / the enforcement
    decoration path, outside the beartype-patch surface — see
    ``.beads/artifacts/mro-31mj/fix-waves/L0-beartype``.
    """

    BEARTYPE_CLAW_SKIP_PACKAGES: ClassVar[FlextTypingBase.VariadicTuple[str]] = (
        "flext_core._models.context",
        "flext_core._typings",
        "flext_core._utilities.logging_config",
        "flext_core._utilities.parser",
        "flext_core._utilities.reliability",
        "flext_core.loggings",
        "flext_core.runtime",
    )
    """Package paths skipped by the flext_core beartype bootstrap."""

    ENFORCEMENT_FORBIDDEN_COLLECTIONS: ClassVar[Mapping[type, str]] = MappingProxyType({
        dict: "Mapping[K, V] or FlextTypingBase.JsonMapping",
        list: "Sequence[X] or FlextTypingBase.JsonList",
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

    ENFORCEMENT_MUTABLE_RUNTIME_TYPES: ClassVar[FlextTypingBase.VariadicTuple[type]] = (
        tuple(
            ENFORCEMENT_FORBIDDEN_COLLECTIONS,
        )
    )
    """Derived view: concrete types used by ``isinstance`` checks."""


__all__: list[str] = ["FlextConstantsEnforcementRuntime"]
