"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import FlextUtilitiesLogging
from flext_core.runtime import FlextRuntime

from ._models.namespace import FlextModelsNamespace
from ._utilities.args import FlextUtilitiesArgs
from ._utilities.beartype_conf import FlextUtilitiesBeartypeConf
from ._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
from ._utilities.checker import FlextUtilitiesChecker
from ._utilities.collection import FlextUtilitiesCollection
from ._utilities.config import FlextUtilitiesConfig
from ._utilities.context import FlextUtilitiesContext
from ._utilities.conversion import FlextUtilitiesConversion
from ._utilities.discovery import FlextUtilitiesDiscovery
from ._utilities.domain import FlextUtilitiesDomain
from ._utilities.enforcement import FlextUtilitiesEnforcement
from ._utilities.enum import FlextUtilitiesEnum
from ._utilities.generators import FlextUtilitiesGenerators
from ._utilities.guards import FlextUtilitiesGuards
from ._utilities.mapper import FlextUtilitiesMapper
from ._utilities.model_runtime import FlextUtilitiesModelRuntime
from ._utilities.parser import FlextUtilitiesParser
from ._utilities.project_metadata import FlextUtilitiesProjectMetadata
from ._utilities.pydantic import FlextUtilitiesPydantic
from ._utilities.reliability import FlextUtilitiesReliability
from ._utilities.settings import FlextUtilitiesSettings
from ._utilities.text import FlextUtilitiesText


class FlextUtilities(
    FlextUtilitiesLogging,
    FlextRuntime,
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesConfig,
    FlextUtilitiesSettings,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
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


u = FlextUtilities

__all__: list[str] = ["FlextUtilities", "u"]
