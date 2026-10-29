"""Fixed project-metadata constants.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from flext_core import t


class FlextConstantsProjectMetadata:
    """Fixed project-metadata constants exposed flat on `c.*`."""

    # NOTE (multi-agent, mro-wkii.17.23 / agent: uv_overlay_owner): immutable
    # pairs replace the model-less mapping while retaining the naming policy.
    SPECIAL_NAME_OVERRIDES: Final[t.StrPairTuple] = (
        ("flext", "FlextRoot"),
        ("flext-core", "Flext"),
    )
    PROJECT_VERSION_PLACEHOLDER: Final[str] = "0.0.0"
    METADATA_SCHEMA_VERSION_DEFAULT: Final[str] = "1.0.0"

    # Family surface: runtime facts the family-surface derivation consumes.
    NAMESPACE_LAYER_NAMES: Final[tuple[str, ...]] = (
        "Constants",
        "Models",
        "Protocols",
        "Types",
        "Utilities",
    )
    """The five canonical facade layers (SSOT seed for every layer derivation)."""

    NAMESPACE_FAMILY_PREFIX: Final[str] = "flext_"
    """Declared family namespace prefix (import-name grammar).

    Discovery seed for runtime family-surface derivation: candidate
    distributions whose normalized name starts with this prefix are family
    members when their root publishes the lazy export contract. The prefix
    narrows discovery only; membership is proven by the published contract,
    never by an enumerated roster.
    """

    FAMILY_SURFACE_MIN_PUBLISHED: Final[int] = 1
    """Minimum family roots that must publish the lazy export contract.

    A runtime where no distribution publishes the contract is a broken
    installation and fails loud instead of deriving an empty surface.
    """

    FAMILY_CANONICAL_ALIASES: Final[frozenset[str]] = frozenset({
        "c",
        "m",
        "p",
        "t",
        "u",
        "d",
        "e",
        "h",
        "r",
        "s",
        "x",
    })
    """Canonical short aliases exposed by FLEXT facade namespaces."""


__all__: tuple[str, ...] = ("FlextConstantsProjectMetadata",)
