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
from typing import TYPE_CHECKING as _TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if _TYPE_CHECKING:
    from flext_core import FlextTypes
    from flext_core._utilities import (
        args,
        cache,
        checker,
        collection,
        configuration,
        context,
        conversion,
        discovery,
        domain,
        enum,
        file_ops,
        generators,
        guards,
        guards_ensure,
        guards_type,
        guards_type_core,
        guards_type_model,
        guards_type_protocol,
        mapper,
        model,
        parser,
        reliability,
        result_helpers,
        text,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs
    from flext_core._utilities.cache import FlextUtilitiesCache
    from flext_core._utilities.checker import FlextUtilitiesChecker
    from flext_core._utilities.collection import FlextUtilitiesCollection
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration
    from flext_core._utilities.context import FlextUtilitiesContext
    from flext_core._utilities.conversion import FlextUtilitiesConversion
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery
    from flext_core._utilities.domain import FlextUtilitiesDomain
    from flext_core._utilities.enum import FlextUtilitiesEnum
    from flext_core._utilities.file_ops import FlextUtilitiesFileOps
    from flext_core._utilities.generators import FlextUtilitiesGenerators
    from flext_core._utilities.guards import FlextUtilitiesGuards
    from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure
    from flext_core._utilities.guards_type import FlextUtilitiesGuardsType
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )
    from flext_core._utilities.mapper import FlextUtilitiesMapper
    from flext_core._utilities.model import FlextUtilitiesModel
    from flext_core._utilities.parser import FlextUtilitiesParser
    from flext_core._utilities.reliability import FlextUtilitiesReliability
    from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers
    from flext_core._utilities.text import FlextUtilitiesText

_LAZY_IMPORTS: FlextTypes.LazyImportIndex = {
    "FlextUtilitiesArgs": "flext_core._utilities.args",
    "FlextUtilitiesCache": "flext_core._utilities.cache",
    "FlextUtilitiesChecker": "flext_core._utilities.checker",
    "FlextUtilitiesCollection": "flext_core._utilities.collection",
    "FlextUtilitiesConfiguration": "flext_core._utilities.configuration",
    "FlextUtilitiesContext": "flext_core._utilities.context",
    "FlextUtilitiesConversion": "flext_core._utilities.conversion",
    "FlextUtilitiesDiscovery": "flext_core._utilities.discovery",
    "FlextUtilitiesDomain": "flext_core._utilities.domain",
    "FlextUtilitiesEnum": "flext_core._utilities.enum",
    "FlextUtilitiesFileOps": "flext_core._utilities.file_ops",
    "FlextUtilitiesGenerators": "flext_core._utilities.generators",
    "FlextUtilitiesGuards": "flext_core._utilities.guards",
    "FlextUtilitiesGuardsEnsure": "flext_core._utilities.guards_ensure",
    "FlextUtilitiesGuardsType": "flext_core._utilities.guards_type",
    "FlextUtilitiesGuardsTypeCore": "flext_core._utilities.guards_type_core",
    "FlextUtilitiesGuardsTypeModel": "flext_core._utilities.guards_type_model",
    "FlextUtilitiesGuardsTypeProtocol": "flext_core._utilities.guards_type_protocol",
    "FlextUtilitiesMapper": "flext_core._utilities.mapper",
    "FlextUtilitiesModel": "flext_core._utilities.model",
    "FlextUtilitiesParser": "flext_core._utilities.parser",
    "FlextUtilitiesReliability": "flext_core._utilities.reliability",
    "FlextUtilitiesResultHelpers": "flext_core._utilities.result_helpers",
    "FlextUtilitiesText": "flext_core._utilities.text",
    "args": "flext_core._utilities.args",
    "cache": "flext_core._utilities.cache",
    "checker": "flext_core._utilities.checker",
    "collection": "flext_core._utilities.collection",
    "configuration": "flext_core._utilities.configuration",
    "context": "flext_core._utilities.context",
    "conversion": "flext_core._utilities.conversion",
    "discovery": "flext_core._utilities.discovery",
    "domain": "flext_core._utilities.domain",
    "enum": "flext_core._utilities.enum",
    "file_ops": "flext_core._utilities.file_ops",
    "generators": "flext_core._utilities.generators",
    "guards": "flext_core._utilities.guards",
    "guards_ensure": "flext_core._utilities.guards_ensure",
    "guards_type": "flext_core._utilities.guards_type",
    "guards_type_core": "flext_core._utilities.guards_type_core",
    "guards_type_model": "flext_core._utilities.guards_type_model",
    "guards_type_protocol": "flext_core._utilities.guards_type_protocol",
    "mapper": "flext_core._utilities.mapper",
    "model": "flext_core._utilities.model",
    "parser": "flext_core._utilities.parser",
    "reliability": "flext_core._utilities.reliability",
    "result_helpers": "flext_core._utilities.result_helpers",
    "text": "flext_core._utilities.text",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
