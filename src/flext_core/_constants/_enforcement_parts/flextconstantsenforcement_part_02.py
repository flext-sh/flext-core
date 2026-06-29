"""Runtime enforcement constants for FlextConstantsEnforcement."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from flext_core._constants._enforcement_parts.flextconstantsenforcement_part_01 import (
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

    Override at process start with the ``BEARTYPE_MODE`` env var
    (``strict`` / ``warn`` / ``off``). Default is ``off`` to keep regular
    runs free of runtime overhead; CI / strict gates set ``strict``.
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
