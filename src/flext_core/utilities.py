"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._models.namespace import FlextModelsNamespace
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
from flext_core._utilities.cache import FlextUtilitiesCache
from flext_core._utilities.checker import FlextUtilitiesChecker
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.conversion import FlextUtilitiesConversion
from flext_core._utilities.discovery import FlextUtilitiesDiscovery
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enforcement import FlextUtilitiesEnforcement
from flext_core._utilities.enum import FlextUtilitiesEnum
from flext_core._utilities.file_ops import FlextUtilitiesFileOps
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.pydantic import FlextUtilitiesPydantic
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers
from flext_core._utilities.settings import FlextUtilitiesSettings
from flext_core._utilities.text import FlextUtilitiesText
from flext_core.loggings import FlextLogger


class FlextUtilities(
    FlextLogger,
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesCache,
    FlextUtilitiesChecker,
    FlextUtilitiesCollection,
    FlextUtilitiesSettings,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDomain,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnum,
    FlextUtilitiesFileOps,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesMapper,
    FlextUtilitiesModel,
    FlextUtilitiesParser,
    FlextUtilitiesPydantic,
    FlextUtilitiesReliability,
    FlextUtilitiesResultHelpers,
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
