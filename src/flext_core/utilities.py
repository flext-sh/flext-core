"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models import FlextModelsNamespace
from flext_core._utilities import (
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesConfig,
    FlextUtilitiesConsole,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesFamilySurface,
    FlextUtilitiesFiles,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesLogging,
    FlextUtilitiesLoggingContext,
    FlextUtilitiesMapper,
    FlextUtilitiesModelRuntime,
    FlextUtilitiesParser,
    FlextUtilitiesProjectMetadata,
    FlextUtilitiesPydantic,
    FlextUtilitiesReliability,
    FlextUtilitiesRuntimeViolationRegistry,
    FlextUtilitiesSettings,
    FlextUtilitiesText,
)
from flext_core.runtime import FlextRuntime


class FlextUtilities(
    FlextUtilitiesLogging,
    FlextRuntime,
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesConsole,
    FlextUtilitiesConfig,
    FlextUtilitiesSettings,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesFamilySurface,
    FlextUtilitiesFiles,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesMapper,
    FlextUtilitiesModelRuntime,
    FlextUtilitiesParser,
    FlextUtilitiesProjectMetadata,
    FlextUtilitiesPydantic,
    FlextUtilitiesReliability,
    FlextUtilitiesRuntimeViolationRegistry,
    FlextUtilitiesText,
    FlextModelsNamespace,
    FlextUtilitiesLoggingContext,
):
    """Unified facade for all FLEXT utility functionality.

    Runtime alias u exposes a flat namespace directly via inheritance.
    Use direct methods only: u.get, u.parse, u.map, u.batch, u.extract,
    u.warn_once, etc. No subdivided namespaces (no u.* at call sites).
    Subprojects use their project u. Aliases/namespaces: MRO registration protocol only.
    """


u = FlextUtilities

__all__: list[str] = ["FlextUtilities", "FlextUtilitiesRuntimeViolationRegistry", "u"]
