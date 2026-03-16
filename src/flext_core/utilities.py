"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import override

from flext_core import FlextRuntime, t
from flext_core._utilities import (
    FlextUtilitiesArgs,
    FlextUtilitiesCache,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesConfiguration,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDeprecation,
    FlextUtilitiesDomain,
    FlextUtilitiesEnum,
    FlextUtilitiesFileOps,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesMapper,
    FlextUtilitiesModel,
    FlextUtilitiesPagination,
    FlextUtilitiesParser,
    FlextUtilitiesReliability,
    FlextUtilitiesText,
    ResultHelpers as FlextUtilitiesResultHelpers,
    validate_pydantic_model,
)


class FlextUtilities(
    FlextRuntime,
    FlextUtilitiesArgs,
    FlextUtilitiesCache,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesConfiguration,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDeprecation,
    FlextUtilitiesDomain,
    FlextUtilitiesEnum,
    FlextUtilitiesFileOps,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesMapper,
    FlextUtilitiesModel,
    FlextUtilitiesPagination,
    FlextUtilitiesParser,
    FlextUtilitiesReliability,
    FlextUtilitiesResultHelpers,
    FlextUtilitiesText,
):
    """Unified facade for all FLEXT utility functionality.

    Runtime alias u exposes a flat namespace directly via inheritance.
    Use direct methods only: u.get, u.parse, u.map, u.from_kwargs, u.batch, u.extract,
    u.warn_once, etc. No subdivided namespaces (no u.* at call sites).
    Subprojects use their project u. Aliases/namespaces: MRO registration protocol only.

    Usage:
        from flext_core import u
        result = u.parse(value, int)
        value = u.get(data, "key")
        mapped = u.map(items, fn)
    """

    @staticmethod
    @override
    def empty(items: t.NormalizedValue | None) -> bool:
        return FlextUtilitiesResultHelpers.empty(items)


u = FlextUtilities
__all__ = ["FlextUtilities", "u", "validate_pydantic_model"]
