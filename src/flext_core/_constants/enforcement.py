"""Enforcement constants for Pydantic v2 runtime governance.

Constants used by FlextModelsBase.Enforcement to validate
class definitions at import time. Accessed via c.ENFORCEMENT_*.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final


class FlextConstantsEnforcement:
    """Constants governing Pydantic v2 enforcement behavior."""

    ENFORCEMENT_MODE: Final[str] = "warn"
    """Controls behavior: "strict" (TypeError), "warn" (UserWarning), "off"."""

    ENFORCEMENT_EXEMPT_MODULE_FRAGMENTS: Final[tuple[str, ...]] = (
        "tests.",
        "test_",
        "conftest",
        "examples.",
        "scripts.",
        "_utilities/adapters",
        "adapters.py",
    )
    """Module path fragments that auto-exempt classes from enforcement."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: Final[frozenset[str]] = frozenset({
        "DynamicModel",
        "FlexibleModel",
        "FlexibleInternalModel",
        "FrozenDynamicModel",
    })
    """Base model names allowed to have relaxed extra= policies."""

    ENFORCEMENT_INFRASTRUCTURE_BASES: Final[frozenset[str]] = frozenset({
        "ManagedModel",
        "EnumManagedModel",
        "NormalizedModel",
        "StrictManagedModel",
        "ArbitraryTypesModel",
        "StrictBoundaryModel",
        "FlexibleInternalModel",
        "ImmutableValueModel",
        "TaggedModel",
        "FlexibleModel",
        "ContractModel",
        "FrozenValueModel",
        "MutableConfiguredMixin",
        "NormalizedMutableConfiguredMixin",
        "Metadata",
        "TimestampableMixin",
        "VersionableMixin",
        "IdentifiableMixin",
        "TimestampedModel",
        "RetryConfigurationMixin",
        "ValidOutcome",
        "InvalidOutcome",
        "WarningOutcome",
    })
    """FLEXT infrastructure base class names exempt from enforcement checks."""

    ENFORCEMENT_FORBIDDEN_COLLECTION_ORIGINS: Final[frozenset[str]] = frozenset({
        "dict",
        "list",
        "set",
    })
    """Collection type names forbidden as field annotation origins."""

    ENFORCEMENT_COLLECTION_REPLACEMENTS: Final[tuple[tuple[str, str], ...]] = (
        ("dict", "Mapping[K, V] or t.ContainerMapping"),
        ("list", "Sequence[X] or t.ContainerList"),
        ("set", "frozenset[X] or AbstractSet[X]"),
    )
    """Replacement suggestions for forbidden collection origins."""

    # --- Constants layer enforcement ---

    ENFORCEMENT_CONSTANTS_SKIP_ATTRS: Final[frozenset[str]] = frozenset({
        "__module__",
        "__qualname__",
        "__doc__",
        "__dict__",
        "__weakref__",
        "__init_subclass__",
        "__subclasshook__",
        "__abstractmethods__",
        "__orig_bases__",
        "__class_getitem__",
        "__pydantic_complete__",
    })
    """Class-level attributes to skip during constants enforcement."""

    ENFORCEMENT_CONSTANTS_FACADE_PREFIXES: Final[tuple[str, ...]] = ("FlextConstants",)
    """Class name prefixes recognized as constants facades."""

    # --- Protocols layer enforcement ---

    ENFORCEMENT_PROTOCOL_EXEMPT_METHODS: Final[frozenset[str]] = frozenset()
    """Methods on protocols classes exempt from protocol-inner-class checks."""

    ENFORCEMENT_PROTOCOL_FACADE_PREFIXES: Final[tuple[str, ...]] = ("FlextProtocols",)
    """Class name prefixes recognized as protocols facades."""

    # --- Types layer enforcement ---

    ENFORCEMENT_TYPES_FACADE_PREFIXES: Final[tuple[str, ...]] = (
        "FlextTypes",
        "FlextTyping",
    )
    """Class name prefixes recognized as types facades."""

    # --- Utilities layer enforcement ---

    ENFORCEMENT_UTILITIES_FACADE_PREFIXES: Final[tuple[str, ...]] = (
        "FlextUtilities",
        "FlextLogger",
    )
    """Class name prefixes recognized as utilities facades."""

    ENFORCEMENT_UTILITIES_EXEMPT_METHODS: Final[frozenset[str]] = frozenset({
        "__init__",
        "__init_subclass__",
        "__new__",
        "__class_getitem__",
    })
    """Methods exempt from static/classmethod enforcement on utilities."""

    # --- MRO Namespace enforcement ---

    ENFORCEMENT_NAMESPACE_MODE: Final[str] = "warn"
    """Separate mode for namespace checks: "strict" / "warn" / "off"."""

    ENFORCEMENT_NAMESPACE_SPECIAL_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
        ("flext_core", "Flext"),
        ("flext", "FlextRoot"),
    )
    """Package→prefix overrides. flext_core→Flext, flext→FlextRoot."""

    ENFORCEMENT_NAMESPACE_FACADE_ROOTS: Final[frozenset[str]] = frozenset({
        "FlextConstants",
        "FlextProtocols",
        "FlextTypes",
        "FlextUtilities",
        "FlextModels",
        "FlextModelsBase",
        "FlextModelsNamespace",
        "EnforcedModel",
    })
    """Root facade class names — skip namespace prefix check on these."""

    ENFORCEMENT_NAMESPACE_LAYER_MAP: Final[tuple[tuple[str, str], ...]] = (
        ("Constants", "constants"),
        ("Models", "models"),
        ("Protocols", "protocols"),
        ("Types", "types"),
        ("Utilities", "utilities"),
    )
    """Class name suffix → layer name mapping for cross-layer detection."""

    ENFORCEMENT_NAMESPACE_STRENUM_ALLOWED_LAYERS: Final[frozenset[str]] = frozenset({
        "constants",
    })
    """Layers where StrEnum inner classes are allowed."""

    ENFORCEMENT_NAMESPACE_PROTOCOL_ALLOWED_LAYERS: Final[frozenset[str]] = frozenset({
        "protocols",
    })
    """Layers where Protocol inner classes are allowed."""
