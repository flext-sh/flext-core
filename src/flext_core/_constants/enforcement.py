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
        "_utilities/adapters",
        "adapters.py",
    )
    """Module path fragments that auto-exempt classes from enforcement."""

    ENFORCEMENT_RELAXED_EXTRA_BASES: Final[frozenset[str]] = frozenset({
        "FlexibleModel",
        "FlexibleInternalModel",
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
