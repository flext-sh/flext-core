"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import (
    FlextLogger,
    FlextModelsNamespace,
    FlextRuntime,
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesMapper,
    FlextUtilitiesParser,
    FlextUtilitiesProjectMetadata,
    FlextUtilitiesPydantic,
    FlextUtilitiesReliability,
    FlextUtilitiesSettings,
    FlextUtilitiesText,
)
from flext_core._utilities.inspect_helpers import FlextUtilitiesInspectHelpers
from flext_core._utilities.model_runtime import FlextUtilitiesModelRuntime


class FlextUtilities(
    FlextLogger,
    FlextRuntime,
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesSettings,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesInspectHelpers,
    FlextUtilitiesMapper,
    FlextUtilitiesModelRuntime,
    FlextUtilitiesParser,
    FlextUtilitiesProjectMetadata,
    FlextUtilitiesPydantic,
    FlextUtilitiesReliability,
    FlextUtilitiesText,
    FlextModelsNamespace,
):
    """Unified facade for all FLEXT utility functionality.

    Runtime alias u exposes a flat namespace directly via inheritance.
    Use direct methods only: u.get, u.parse, u.map, u.batch, u.extract,
    u.warn_once, etc. No subdivided namespaces (no u.* at call sites).
    Subprojects use their project u. Aliases/namespaces: MRO registration protocol only.
    """


__all__: list[str] = ["FlextUtilities", "u"]

u = FlextUtilities
