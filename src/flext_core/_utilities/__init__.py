# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Internal utilities package - implementation modules only.

This package contains implementation modules for u.
External code MUST NOT import from this package directly.
Use u from flext_core instead.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._utilities import (
        args as args,
        cache as cache,
        checker as checker,
        collection as collection,
        configuration as configuration,
        context as context,
        conversion as conversion,
        deprecation as deprecation,
        discovery as discovery,
        domain as domain,
        enum as enum,
        file_ops as file_ops,
        generators as generators,
        guards as guards,
        guards_ensure as guards_ensure,
        guards_type as guards_type,
        guards_type_core as guards_type_core,
        guards_type_model as guards_type_model,
        guards_type_protocol as guards_type_protocol,
        mapper as mapper,
        model as model,
        pagination as pagination,
        parser as parser,
        reliability as reliability,
        result_helpers as result_helpers,
        text as text,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs as FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache as FlextUtilitiesCache
    from flext_core._utilities.checker import (
        FlextUtilitiesChecker as FlextUtilitiesChecker,
    )
    from flext_core._utilities.collection import (
        FlextUtilitiesCollection as FlextUtilitiesCollection,
    )
    from flext_core._utilities.configuration import (
        FlextUtilitiesConfiguration as FlextUtilitiesConfiguration,
    )
    from flext_core._utilities.context import (
        FlextUtilitiesContext as FlextUtilitiesContext,
    )
    from flext_core._utilities.conversion import (
        FlextUtilitiesConversion as FlextUtilitiesConversion,
    )
    from flext_core._utilities.discovery import (
        FlextUtilitiesDiscovery as FlextUtilitiesDiscovery,
    )
    from flext_core._utilities.domain import (
        FlextUtilitiesDomain as FlextUtilitiesDomain,
    )
    from flext_core._utilities.enum import FlextUtilitiesEnum as FlextUtilitiesEnum
    from flext_core._utilities.file_ops import (
        FlextUtilitiesFileOps as FlextUtilitiesFileOps,
    )
    from flext_core._utilities.generators import (
        FlextUtilitiesGenerators as FlextUtilitiesGenerators,
    )
    from flext_core._utilities.guards import (
        FlextUtilitiesGuards as FlextUtilitiesGuards,
    )
    from flext_core._utilities.guards_ensure import (
        FlextUtilitiesGuardsEnsure as FlextUtilitiesGuardsEnsure,
    )
    from flext_core._utilities.guards_type import (
        FlextUtilitiesGuardsType as FlextUtilitiesGuardsType,
    )
    from flext_core._utilities.guards_type_core import (
        FlextUtilitiesGuardsTypeCore as FlextUtilitiesGuardsTypeCore,
    )
    from flext_core._utilities.guards_type_model import (
        FlextUtilitiesGuardsTypeModel as FlextUtilitiesGuardsTypeModel,
    )
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol as FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.mapper import (
        FlextUtilitiesMapper as FlextUtilitiesMapper,
    )
    from flext_core._utilities.model import FlextUtilitiesModel as FlextUtilitiesModel
    from flext_core._utilities.pagination import (
        FlextUtilitiesPagination as FlextUtilitiesPagination,
    )
    from flext_core._utilities.parser import (
        FlextUtilitiesParser as FlextUtilitiesParser,
    )
    from flext_core._utilities.reliability import (
        FlextUtilitiesReliability as FlextUtilitiesReliability,
    )
    from flext_core._utilities.result_helpers import (
        FlextUtilitiesResultHelpers as FlextUtilitiesResultHelpers,
    )
    from flext_core._utilities.text import FlextUtilitiesText as FlextUtilitiesText

_LAZY_IMPORTS: Mapping[str, Sequence[str]] = {
    "FlextUtilitiesArgs": ["flext_core._utilities.args", "FlextUtilitiesArgs"],
    "FlextUtilitiesCache": ["flext_core._utilities.cache", "FlextUtilitiesCache"],
    "FlextUtilitiesChecker": ["flext_core._utilities.checker", "FlextUtilitiesChecker"],
    "FlextUtilitiesCollection": [
        "flext_core._utilities.collection",
        "FlextUtilitiesCollection",
    ],
    "FlextUtilitiesConfiguration": [
        "flext_core._utilities.configuration",
        "FlextUtilitiesConfiguration",
    ],
    "FlextUtilitiesContext": ["flext_core._utilities.context", "FlextUtilitiesContext"],
    "FlextUtilitiesConversion": [
        "flext_core._utilities.conversion",
        "FlextUtilitiesConversion",
    ],
    "FlextUtilitiesDiscovery": [
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
    ],
    "FlextUtilitiesDomain": ["flext_core._utilities.domain", "FlextUtilitiesDomain"],
    "FlextUtilitiesEnum": ["flext_core._utilities.enum", "FlextUtilitiesEnum"],
    "FlextUtilitiesFileOps": [
        "flext_core._utilities.file_ops",
        "FlextUtilitiesFileOps",
    ],
    "FlextUtilitiesGenerators": [
        "flext_core._utilities.generators",
        "FlextUtilitiesGenerators",
    ],
    "FlextUtilitiesGuards": ["flext_core._utilities.guards", "FlextUtilitiesGuards"],
    "FlextUtilitiesGuardsEnsure": [
        "flext_core._utilities.guards_ensure",
        "FlextUtilitiesGuardsEnsure",
    ],
    "FlextUtilitiesGuardsType": [
        "flext_core._utilities.guards_type",
        "FlextUtilitiesGuardsType",
    ],
    "FlextUtilitiesGuardsTypeCore": [
        "flext_core._utilities.guards_type_core",
        "FlextUtilitiesGuardsTypeCore",
    ],
    "FlextUtilitiesGuardsTypeModel": [
        "flext_core._utilities.guards_type_model",
        "FlextUtilitiesGuardsTypeModel",
    ],
    "FlextUtilitiesGuardsTypeProtocol": [
        "flext_core._utilities.guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocol",
    ],
    "FlextUtilitiesMapper": ["flext_core._utilities.mapper", "FlextUtilitiesMapper"],
    "FlextUtilitiesModel": ["flext_core._utilities.model", "FlextUtilitiesModel"],
    "FlextUtilitiesPagination": [
        "flext_core._utilities.pagination",
        "FlextUtilitiesPagination",
    ],
    "FlextUtilitiesParser": ["flext_core._utilities.parser", "FlextUtilitiesParser"],
    "FlextUtilitiesReliability": [
        "flext_core._utilities.reliability",
        "FlextUtilitiesReliability",
    ],
    "FlextUtilitiesResultHelpers": [
        "flext_core._utilities.result_helpers",
        "FlextUtilitiesResultHelpers",
    ],
    "FlextUtilitiesText": ["flext_core._utilities.text", "FlextUtilitiesText"],
    "args": ["flext_core._utilities.args", ""],
    "cache": ["flext_core._utilities.cache", ""],
    "checker": ["flext_core._utilities.checker", ""],
    "collection": ["flext_core._utilities.collection", ""],
    "configuration": ["flext_core._utilities.configuration", ""],
    "context": ["flext_core._utilities.context", ""],
    "conversion": ["flext_core._utilities.conversion", ""],
    "deprecation": ["flext_core._utilities.deprecation", ""],
    "discovery": ["flext_core._utilities.discovery", ""],
    "domain": ["flext_core._utilities.domain", ""],
    "enum": ["flext_core._utilities.enum", ""],
    "file_ops": ["flext_core._utilities.file_ops", ""],
    "generators": ["flext_core._utilities.generators", ""],
    "guards": ["flext_core._utilities.guards", ""],
    "guards_ensure": ["flext_core._utilities.guards_ensure", ""],
    "guards_type": ["flext_core._utilities.guards_type", ""],
    "guards_type_core": ["flext_core._utilities.guards_type_core", ""],
    "guards_type_model": ["flext_core._utilities.guards_type_model", ""],
    "guards_type_protocol": ["flext_core._utilities.guards_type_protocol", ""],
    "mapper": ["flext_core._utilities.mapper", ""],
    "model": ["flext_core._utilities.model", ""],
    "pagination": ["flext_core._utilities.pagination", ""],
    "parser": ["flext_core._utilities.parser", ""],
    "reliability": ["flext_core._utilities.reliability", ""],
    "result_helpers": ["flext_core._utilities.result_helpers", ""],
    "text": ["flext_core._utilities.text", ""],
}

_EXPORTS: Sequence[str] = [
    "FlextUtilitiesArgs",
    "FlextUtilitiesCache",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnum",
    "FlextUtilitiesFileOps",
    "FlextUtilitiesGenerators",
    "FlextUtilitiesGuards",
    "FlextUtilitiesGuardsEnsure",
    "FlextUtilitiesGuardsType",
    "FlextUtilitiesGuardsTypeCore",
    "FlextUtilitiesGuardsTypeModel",
    "FlextUtilitiesGuardsTypeProtocol",
    "FlextUtilitiesMapper",
    "FlextUtilitiesModel",
    "FlextUtilitiesPagination",
    "FlextUtilitiesParser",
    "FlextUtilitiesReliability",
    "FlextUtilitiesResultHelpers",
    "FlextUtilitiesText",
    "args",
    "cache",
    "checker",
    "collection",
    "configuration",
    "context",
    "conversion",
    "deprecation",
    "discovery",
    "domain",
    "enum",
    "file_ops",
    "generators",
    "guards",
    "guards_ensure",
    "guards_type",
    "guards_type_core",
    "guards_type_model",
    "guards_type_protocol",
    "mapper",
    "model",
    "pagination",
    "parser",
    "reliability",
    "result_helpers",
    "text",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, _EXPORTS)
