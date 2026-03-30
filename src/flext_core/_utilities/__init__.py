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
    from flext_core._utilities.args import *
    from flext_core._utilities.cache import *
    from flext_core._utilities.checker import *
    from flext_core._utilities.collection import *
    from flext_core._utilities.configuration import *
    from flext_core._utilities.context import *
    from flext_core._utilities.conversion import *
    from flext_core._utilities.discovery import *
    from flext_core._utilities.domain import *
    from flext_core._utilities.enum import *
    from flext_core._utilities.file_ops import *
    from flext_core._utilities.generators import *
    from flext_core._utilities.guards import *
    from flext_core._utilities.guards_ensure import *
    from flext_core._utilities.guards_type import *
    from flext_core._utilities.guards_type_core import *
    from flext_core._utilities.guards_type_model import *
    from flext_core._utilities.guards_type_protocol import *
    from flext_core._utilities.mapper import *
    from flext_core._utilities.model import *
    from flext_core._utilities.pagination import *
    from flext_core._utilities.parser import *
    from flext_core._utilities.reliability import *
    from flext_core._utilities.result_helpers import *
    from flext_core._utilities.text import *

_LAZY_IMPORTS: Mapping[str, str | Sequence[str]] = {
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
    "FlextUtilitiesPagination": "flext_core._utilities.pagination",
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
    "deprecation": "flext_core._utilities.deprecation",
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
    "pagination": "flext_core._utilities.pagination",
    "parser": "flext_core._utilities.parser",
    "reliability": "flext_core._utilities.reliability",
    "result_helpers": "flext_core._utilities.result_helpers",
    "text": "flext_core._utilities.text",
}


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
