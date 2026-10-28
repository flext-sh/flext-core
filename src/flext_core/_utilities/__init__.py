# AUTO-GENERATED FILE — Regenerate with: make gen
"""Internal package for utilities.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities._logging_context.bound_logger import (
        FlextUtilitiesLoggingBoundLogger,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.base import FlextUtilitiesBase
    from flext_core._utilities.checker import FlextUtilitiesChecker
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
    from flext_core._utilities.enums import FlextUtilitiesEnum
    from flext_core._utilities.family_surface import FlextUtilitiesFamilySurface
    from flext_core._utilities.files import FlextUtilitiesFiles
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.handler import FlextUtilitiesHandler
    from flext_core._utilities.logging import FlextUtilitiesLogging
    from flext_core._utilities.logging_config import FlextUtilitiesLoggingConfig
    from flext_core._utilities.logging_context import FlextUtilitiesLoggingContext
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.mapper_access import FlextUtilitiesMapperAccess
    from flext_core._utilities.mapper_extract import FlextUtilitiesMapperExtract
    from flext_core._utilities.metadata import FlextUtilitiesMetadata
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.model_options import FlextUtilitiesModelOptions
    from flext_core._utilities.model_runtime import FlextUtilitiesModelRuntime
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.parser_coerce import FlextUtilitiesParserCoerce
    from flext_core._utilities.parser_targets import FlextUtilitiesParserTargets
    from flext_core._utilities.project_metadata import FlextUtilitiesProjectMetadata
    from flext_core._utilities.pydantic import FlextUtilitiesPydantic
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.settings import FlextUtilitiesSettings
    from flext_core._utilities.text import FlextUtilitiesText
    from flext_core._utilities.typeadapter_json import FlextUtilitiesTypingAdapterJson
    from flext_core._utilities.typeadapter_scalars import (
        FlextUtilitiesTypingAdapterScalars,
    )

__all__: tuple[str, ...] = (
    "FlextUtilitiesArgs",
    "FlextUtilitiesBase",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesCollectionIter",
    "FlextUtilitiesCollectionMerge",
    "FlextUtilitiesConfig",
    "FlextUtilitiesConsole",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDispatcherExecute",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnum",
    "FlextUtilitiesFamilySurface",
    "FlextUtilitiesFiles",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesHandler",
    "FlextUtilitiesLogging",
    "FlextUtilitiesLoggingBoundLogger",
    "FlextUtilitiesLoggingConfig",
    "FlextUtilitiesLoggingContext",
    "FlextUtilitiesMapper",
    "FlextUtilitiesMapperAccess",
    "FlextUtilitiesMapperExtract",
    "FlextUtilitiesMetadata",
    "FlextUtilitiesModel",
    "FlextUtilitiesModelOptions",
    "FlextUtilitiesModelRuntime",
    "FlextUtilitiesParser",
    "FlextUtilitiesParserCoerce",
    "FlextUtilitiesParserTargets",
    "FlextUtilitiesProjectMetadata",
    "FlextUtilitiesPydantic",
    "FlextUtilitiesReliability",
    "FlextUtilitiesSettings",
    "FlextUtilitiesText",
    "FlextUtilitiesTypingAdapterJson",
    "FlextUtilitiesTypingAdapterScalars",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextUtilitiesLoggingBoundLogger": "._logging_context.bound_logger",
        "FlextUtilitiesArgs": ".args",
        "FlextUtilitiesBase": ".base",
        "FlextUtilitiesChecker": ".checker",
        "FlextUtilitiesCollection": ".collection",
        "FlextUtilitiesCollectionIter": ".collection_iter",
        "FlextUtilitiesCollectionMerge": ".collection_merge",
        "FlextUtilitiesConfig": ".config",
        "FlextUtilitiesConsole": ".console",
        "FlextUtilitiesContext": ".context",
        "FlextUtilitiesConversion": ".conversion",
        "FlextUtilitiesDiscovery": ".discovery",
        "FlextUtilitiesDispatcherExecute": ".dispatcher_execute",
        "FlextUtilitiesDomain": ".domain",
        "FlextUtilitiesEnum": ".enums",
        "FlextUtilitiesFamilySurface": ".family_surface",
        "FlextUtilitiesFiles": ".files",
        "FlextUtilitiesGenerators": ".generators",
        "FlextUtilitiesGuards": ".guards",
        "FlextUtilitiesGuardsTypeCore": ".guards_type_core",
        "FlextUtilitiesGuardsTypeModel": ".guards_type_model",
        "FlextUtilitiesGuardsTypeProtocol": ".guards_type_protocol",
        "FlextUtilitiesHandler": ".handler",
        "FlextUtilitiesLogging": ".logging",
        "FlextUtilitiesLoggingConfig": ".logging_config",
        "FlextUtilitiesLoggingContext": ".logging_context",
        "FlextUtilitiesMapper": ".mapper",
        "FlextUtilitiesMapperAccess": ".mapper_access",
        "FlextUtilitiesMapperExtract": ".mapper_extract",
        "FlextUtilitiesMetadata": ".metadata",
        "FlextUtilitiesModel": ".model",
        "FlextUtilitiesModelOptions": ".model_options",
        "FlextUtilitiesModelRuntime": ".model_runtime",
        "FlextUtilitiesParser": ".parser",
        "FlextUtilitiesParserCoerce": ".parser_coerce",
        "FlextUtilitiesParserTargets": ".parser_targets",
        "FlextUtilitiesProjectMetadata": ".project_metadata",
        "FlextUtilitiesPydantic": ".pydantic",
        "FlextUtilitiesReliability": ".reliability",
        "FlextUtilitiesSettings": ".settings",
        "FlextUtilitiesText": ".text",
        "FlextUtilitiesTypingAdapterJson": ".typeadapter_json",
        "FlextUtilitiesTypingAdapterScalars": ".typeadapter_scalars",
    }),
    public_exports=__all__,
)
