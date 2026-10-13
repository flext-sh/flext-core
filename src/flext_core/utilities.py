"""FlextUtilities - Pure Facade for FLEXT Utility Classes.

Runtime alias u: flat namespace via inheritance from _utilities/* subclasses.
Use u.get, u.parse, u.map, etc. (no u.*). Subprojects use their project u.
Aliases/namespaces: MRO registration protocol only. No local implementations.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core import runtime
from flext_core._models import FlextModelsNamespace
from flext_core._utilities import (
    FlextUtilitiesChecker,
    FlextUtilitiesEnforcement,
    FlextUtilitiesLoggingContext,
)
from flext_core._utilities._beartype import (
    FlextUtilitiesBeartypeImportBlacklistVisitor,
    FlextUtilitiesBeartypeModuleSource,
    FlextUtilitiesBeartypeTypeAliases,
)
from flext_core._utilities.args import FlextUtilitiesArgs
from flext_core._utilities.base import FlextUtilitiesBase
from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf
from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine
from flext_core._utilities.collection import FlextUtilitiesCollection
from flext_core._utilities.collection_iter import FlextUtilitiesCollectionIter
from flext_core._utilities.collection_merge import FlextUtilitiesCollectionMerge
from flext_core._utilities.config import FlextUtilitiesConfig
from flext_core._utilities.console import FlextUtilitiesConsole
from flext_core._utilities.context import FlextUtilitiesContext
from flext_core._utilities.conversion import FlextUtilitiesConversion
from flext_core._utilities.discovery import FlextUtilitiesDiscovery
from flext_core._utilities.dispatcher_execute import FlextUtilitiesDispatcherExecute
from flext_core._utilities.domain import FlextUtilitiesDomain
from flext_core._utilities.enforcement_emit import FlextUtilitiesEnforcementEmit
from flext_core._utilities.enums import FlextUtilitiesEnum
from flext_core._utilities.family_surface import FlextUtilitiesFamilySurface
from flext_core._utilities.files import FlextUtilitiesFiles
from flext_core._utilities.generators import FlextUtilitiesGenerators
from flext_core._utilities.guards import FlextUtilitiesGuards
from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
from flext_core._utilities.guards_type_protocol import FlextUtilitiesGuardsTypeProtocol
from flext_core._utilities._guards_type_protocol_specs import (
    FlextUtilitiesGuardsTypeProtocolSpecsMixin,
)
from flext_core._utilities._guards_type_protocol_string import (
    FlextUtilitiesGuardsTypeProtocolStringMixin,
)
from flext_core._utilities.handler import FlextUtilitiesHandler
from flext_core._utilities.logging import FlextUtilitiesLogging
from flext_core._utilities.mapper import FlextUtilitiesMapper
from flext_core._utilities.model import FlextUtilitiesModel
from flext_core._utilities.model_options import FlextUtilitiesModelOptions
from flext_core._utilities.model_runtime import FlextUtilitiesModelRuntime
from flext_core._utilities.parser import FlextUtilitiesParser
from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce
from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
from flext_core._utilities.pydantic import FlextUtilitiesPydantic
from flext_core._utilities.reliability import FlextUtilitiesReliability
from flext_core._utilities.runtime_violation_registry import (
    FlextUtilitiesRuntimeViolationRegistry,
)
from flext_core._utilities.settings import FlextUtilitiesSettings
from flext_core._utilities.text import FlextUtilitiesText
from flext_core._utilities.typeadapter_json import FlextUtilitiesTypingAdapterJson
from flext_core._utilities.typeadapter_scalars import FlextUtilitiesTypingAdapterScalars


class FlextUtilities(
    FlextUtilitiesArgs,
    FlextUtilitiesBeartypeConf,
    FlextUtilitiesBeartypeEngine,
    FlextUtilitiesCollection,
    FlextUtilitiesCollectionIter,
    FlextUtilitiesCollectionMerge,
    FlextUtilitiesConfig,
    FlextUtilitiesConsole,
    FlextUtilitiesContext,
    FlextUtilitiesConversion,
    FlextUtilitiesDiscovery,
    FlextUtilitiesDispatcherExecute,
    FlextUtilitiesDomain,
    FlextUtilitiesEnum,
    FlextUtilitiesFamilySurface,
    FlextUtilitiesFiles,
    FlextUtilitiesGenerators,
    FlextUtilitiesGuards,
    FlextUtilitiesGuardsTypeCore,
    FlextUtilitiesGuardsTypeModel,
    FlextUtilitiesGuardsTypeProtocol,
    FlextUtilitiesGuardsTypeProtocolStringMixin,
    FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    FlextUtilitiesHandler,
    FlextUtilitiesLogging,
    FlextUtilitiesLoggingContext,
    FlextUtilitiesMapper,
    FlextUtilitiesModelRuntime,
    FlextUtilitiesModelOptions,
    FlextUtilitiesModel,
    FlextUtilitiesParser,
    FlextUtilitiesParserCoerce,
    FlextUtilitiesProjectMetadata,
    FlextUtilitiesPydantic,
    FlextUtilitiesTypingAdapterJson,
    FlextUtilitiesTypingAdapterScalars,
    FlextUtilitiesBase,
    FlextUtilitiesReliability,
    FlextUtilitiesRuntimeViolationRegistry,
    FlextUtilitiesSettings,
    FlextUtilitiesText,
    runtime.FlextRuntime,
    FlextUtilitiesChecker,
    FlextUtilitiesEnforcement,
    FlextUtilitiesEnforcementEmit,
    FlextModelsNamespace,
    FlextUtilitiesBeartypeModuleSource,
    FlextUtilitiesBeartypeTypeAliases,
    FlextUtilitiesBeartypeImportBlacklistVisitor,
):
    """Unified facade for all FLEXT utility functionality.

    Runtime alias u exposes a flat namespace directly via inheritance.
    Use direct methods only: u.get, u.parse, u.map, u.batch, u.extract,
    u.warn_once, etc. No subdivided namespaces (no u.* at call sites).
    Subprojects use their project u. Aliases/namespaces: MRO registration protocol only.
    """


u = FlextUtilities

__all__: list[str] = [
    "FlextUtilities",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesEnforcement",
    "FlextUtilitiesFamilySurface",
    "FlextUtilitiesMapper",
    "FlextUtilitiesRuntimeViolationRegistry",
    "u",
]
