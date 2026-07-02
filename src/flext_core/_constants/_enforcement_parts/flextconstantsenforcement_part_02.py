"""Runtime enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from .flextconstantsenforcement_part_01 import (
    FlextConstantsEnforcementEnums,
)


class FlextConstantsEnforcementRuntime:
    """Runtime modes, base exemptions, and collection contracts."""

    ENFORCEMENT_MODE: Final[FlextConstantsEnforcementEnums.EnforcementMode] = (
        FlextConstantsEnforcementEnums.EnforcementMode.WARN
    )
    """Controls behavior: strict (TypeError), warn (UserWarning), off."""

    BEARTYPE_MODE: Final[FlextConstantsEnforcementEnums.EnforcementMode] = (
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
    nested-class facades (e.g. ``FlextLogger.PerformanceTracker``). Fixing it
    requires changes to beartype's core class decorator / the enforcement
    decoration path, outside the beartype-patch surface — see
    ``.beads/artifacts/mro-31mj/fix-waves/L0-beartype``.
    """

    BEARTYPE_CLAW_SKIP_PACKAGES: Final[tuple[str, ...]] = (
        "flext_core._models.context",
        "flext_core._typings",
        "flext_core._utilities.logging_config",
        "flext_core._utilities.parser",
        "flext_core._utilities.reliability",
        "flext_core.loggings",
        "flext_core.runtime",
    )
    """Package paths skipped by the flext_core beartype bootstrap."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: Final[frozenset[str]] = frozenset({
        "DynamicModel",
        "FlexibleModel",
        "FlexibleInternalModel",
        "FrozenDynamicModel",
    })
    """Base model names allowed to have relaxed extra= policies."""

    ENFORCEMENT_INFRASTRUCTURE_BASES: Final[frozenset[str]] = frozenset({
        "ArbitraryTypesModel",
        "ContractModel",
        "EnumManagedModel",
        "FlexibleInternalModel",
        "FlexibleModel",
        "FrozenValueModel",
        "IdentifiableMixin",
        "ImmutableValueModel",
        "InvalidOutcome",
        "ManagedModel",
        "Metadata",
        "MutableConfiguredMixin",
        "NormalizedModel",
        "NormalizedMutableConfiguredMixin",
        "RetryConfigurationMixin",
        "StrictBoundaryModel",
        "StrictManagedModel",
        "TaggedModel",
        "TimestampableMixin",
        "TimestampedModel",
        "ValidOutcome",
        "VersionableMixin",
        "WarningOutcome",
    })
    """FLEXT infrastructure base class names exempt from enforcement checks."""

    ENFORCEMENT_FORBIDDEN_COLLECTIONS: Final[Mapping[type, str]] = MappingProxyType({
        dict: "Mapping[K, V] or t.JsonMapping",
        list: "Sequence[X] or t.JsonList",
        set: "frozenset[X] or AbstractSet[X]",
    })
    """SSOT: forbidden mutable-collection types mapped to replacement hints.

    Downstream constants (``ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS`` as
    name set, ``ENFORCEMENT_MUTABLE_RUNTIME_TYPES`` as runtime tuple) are
    derived from this single mapping — do not maintain parallel lists.
    """

    ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS: Final[frozenset[str]] = frozenset(
        kind.__name__ for kind in ENFORCEMENT_FORBIDDEN_COLLECTIONS
    )
    """Derived view: collection names used by annotation-origin checks."""

    ENFORCEMENT_MUTABLE_RUNTIME_TYPES: Final[tuple[type, ...]] = tuple(
        ENFORCEMENT_FORBIDDEN_COLLECTIONS,
    )
    """Derived view: concrete types used by ``isinstance`` checks."""

    # --- Per-layer metadata (single SSOT mappings keyed by EnforcementLayer) ---

    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: Final[frozenset[str]] = frozenset({
        "__abstractmethods__",
        "__class_getitem__",
        "__dict__",
        "__doc__",
        "__init_subclass__",
        "__module__",
        "__orig_bases__",
        "__pydantic_complete__",
        "__qualname__",
        "__subclasshook__",
        "__weakref__",
        # Pydantic v2 class-level contract attributes — NOT constants,
        # they are framework metadata owned by the BaseModel machinery.
        "model_computed_fields",
        "model_config",
        "model_extra",
        "model_fields",
        "model_post_init",
    })
    """Class-level attributes to skip during constants enforcement."""

    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: Final[frozenset[str]] = frozenset({
        "__class_getitem__",
        "__init__",
        "__init_subclass__",
        "__new__",
    })
    """Methods exempt from static/classmethod enforcement on utilities."""


__all__: list[str] = ["FlextConstantsEnforcementRuntime"]
