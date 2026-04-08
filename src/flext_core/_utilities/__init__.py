# AUTO-GENERATED FILE — DO NOT EDIT MANUALLY.
# Regenerate with: make gen
#
"""Utilities package."""

from __future__ import annotations

import typing as _t

from flext_core.lazy import install_lazy_exports

if _t.TYPE_CHECKING:
    import flext_core._utilities.args as _flext_core__utilities_args

    args = _flext_core__utilities_args
    import flext_core._utilities.beartype_conf as _flext_core__utilities_beartype_conf
    from flext_core._utilities.args import FlextUtilitiesArgs

    beartype_conf = _flext_core__utilities_beartype_conf
    import flext_core._utilities.beartype_engine as _flext_core__utilities_beartype_engine
    from flext_core._utilities.beartype_conf import FlextUtilitiesBeartypeConf

    beartype_engine = _flext_core__utilities_beartype_engine
    import flext_core._utilities.cache as _flext_core__utilities_cache
    from flext_core._utilities.beartype_engine import FlextUtilitiesBeartypeEngine

    cache = _flext_core__utilities_cache
    import flext_core._utilities.checker as _flext_core__utilities_checker
    from flext_core._utilities.cache import FlextUtilitiesCache

    checker = _flext_core__utilities_checker
    import flext_core._utilities.collection as _flext_core__utilities_collection
    from flext_core._utilities.checker import FlextUtilitiesChecker

    collection = _flext_core__utilities_collection
    import flext_core._utilities.configuration as _flext_core__utilities_configuration
    from flext_core._utilities.collection import FlextUtilitiesCollection

    configuration = _flext_core__utilities_configuration
    import flext_core._utilities.context as _flext_core__utilities_context
    from flext_core._utilities.configuration import FlextUtilitiesConfiguration

    context = _flext_core__utilities_context
    import flext_core._utilities.conversion as _flext_core__utilities_conversion
    from flext_core._utilities.context import FlextUtilitiesContext

    conversion = _flext_core__utilities_conversion
    import flext_core._utilities.discovery as _flext_core__utilities_discovery
    from flext_core._utilities.conversion import FlextUtilitiesConversion

    discovery = _flext_core__utilities_discovery
    import flext_core._utilities.domain as _flext_core__utilities_domain
    from flext_core._utilities.discovery import FlextUtilitiesDiscovery

    domain = _flext_core__utilities_domain
    import flext_core._utilities.enforcement as _flext_core__utilities_enforcement
    from flext_core._utilities.domain import FlextUtilitiesDomain

    enforcement = _flext_core__utilities_enforcement
    import flext_core._utilities.enum as _flext_core__utilities_enum
    from flext_core._utilities.enforcement import FlextUtilitiesEnforcement

    enum = _flext_core__utilities_enum
    import flext_core._utilities.file_ops as _flext_core__utilities_file_ops
    from flext_core._utilities.enum import FlextUtilitiesEnum

    file_ops = _flext_core__utilities_file_ops
    import flext_core._utilities.generators as _flext_core__utilities_generators
    from flext_core._utilities.file_ops import FlextUtilitiesFileOps

    generators = _flext_core__utilities_generators
    import flext_core._utilities.guards as _flext_core__utilities_guards
    from flext_core._utilities.generators import FlextUtilitiesGenerators

    guards = _flext_core__utilities_guards
    import flext_core._utilities.guards_ensure as _flext_core__utilities_guards_ensure
    from flext_core._utilities.guards import FlextUtilitiesGuards

    guards_ensure = _flext_core__utilities_guards_ensure
    import flext_core._utilities.guards_type as _flext_core__utilities_guards_type
    from flext_core._utilities.guards_ensure import FlextUtilitiesGuardsEnsure

    guards_type = _flext_core__utilities_guards_type
    import flext_core._utilities.guards_type_core as _flext_core__utilities_guards_type_core
    from flext_core._utilities.guards_type import FlextUtilitiesGuardsType

    guards_type_core = _flext_core__utilities_guards_type_core
    import flext_core._utilities.guards_type_model as _flext_core__utilities_guards_type_model
    from flext_core._utilities.guards_type_core import FlextUtilitiesGuardsTypeCore

    guards_type_model = _flext_core__utilities_guards_type_model
    import flext_core._utilities.guards_type_protocol as _flext_core__utilities_guards_type_protocol
    from flext_core._utilities.guards_type_model import FlextUtilitiesGuardsTypeModel

    guards_type_protocol = _flext_core__utilities_guards_type_protocol
    import flext_core._utilities.mapper as _flext_core__utilities_mapper
    from flext_core._utilities.guards_type_protocol import (
        FlextUtilitiesGuardsTypeProtocol,
    )

    mapper = _flext_core__utilities_mapper
    import flext_core._utilities.model as _flext_core__utilities_model
    from flext_core._utilities.mapper import FlextUtilitiesMapper

    model = _flext_core__utilities_model
    import flext_core._utilities.parser as _flext_core__utilities_parser
    from flext_core._utilities.model import FlextUtilitiesModel

    parser = _flext_core__utilities_parser
    import flext_core._utilities.reliability as _flext_core__utilities_reliability
    from flext_core._utilities.parser import FlextUtilitiesParser

    reliability = _flext_core__utilities_reliability
    import flext_core._utilities.result_helpers as _flext_core__utilities_result_helpers
    from flext_core._utilities.reliability import FlextUtilitiesReliability

    result_helpers = _flext_core__utilities_result_helpers
    import flext_core._utilities.text as _flext_core__utilities_text
    from flext_core._utilities.result_helpers import FlextUtilitiesResultHelpers

    text = _flext_core__utilities_text
    from flext_core._utilities.text import FlextUtilitiesText
_LAZY_IMPORTS = {
    "FlextUtilitiesArgs": ("flext_core._utilities.args", "FlextUtilitiesArgs"),
    "FlextUtilitiesBeartypeConf": (
        "flext_core._utilities.beartype_conf",
        "FlextUtilitiesBeartypeConf",
    ),
    "FlextUtilitiesBeartypeEngine": (
        "flext_core._utilities.beartype_engine",
        "FlextUtilitiesBeartypeEngine",
    ),
    "FlextUtilitiesCache": ("flext_core._utilities.cache", "FlextUtilitiesCache"),
    "FlextUtilitiesChecker": ("flext_core._utilities.checker", "FlextUtilitiesChecker"),
    "FlextUtilitiesCollection": (
        "flext_core._utilities.collection",
        "FlextUtilitiesCollection",
    ),
    "FlextUtilitiesConfiguration": (
        "flext_core._utilities.configuration",
        "FlextUtilitiesConfiguration",
    ),
    "FlextUtilitiesContext": ("flext_core._utilities.context", "FlextUtilitiesContext"),
    "FlextUtilitiesConversion": (
        "flext_core._utilities.conversion",
        "FlextUtilitiesConversion",
    ),
    "FlextUtilitiesDiscovery": (
        "flext_core._utilities.discovery",
        "FlextUtilitiesDiscovery",
    ),
    "FlextUtilitiesDomain": ("flext_core._utilities.domain", "FlextUtilitiesDomain"),
    "FlextUtilitiesEnforcement": (
        "flext_core._utilities.enforcement",
        "FlextUtilitiesEnforcement",
    ),
    "FlextUtilitiesEnum": ("flext_core._utilities.enum", "FlextUtilitiesEnum"),
    "FlextUtilitiesFileOps": (
        "flext_core._utilities.file_ops",
        "FlextUtilitiesFileOps",
    ),
    "FlextUtilitiesGenerators": (
        "flext_core._utilities.generators",
        "FlextUtilitiesGenerators",
    ),
    "FlextUtilitiesGuards": ("flext_core._utilities.guards", "FlextUtilitiesGuards"),
    "FlextUtilitiesGuardsEnsure": (
        "flext_core._utilities.guards_ensure",
        "FlextUtilitiesGuardsEnsure",
    ),
    "FlextUtilitiesGuardsType": (
        "flext_core._utilities.guards_type",
        "FlextUtilitiesGuardsType",
    ),
    "FlextUtilitiesGuardsTypeCore": (
        "flext_core._utilities.guards_type_core",
        "FlextUtilitiesGuardsTypeCore",
    ),
    "FlextUtilitiesGuardsTypeModel": (
        "flext_core._utilities.guards_type_model",
        "FlextUtilitiesGuardsTypeModel",
    ),
    "FlextUtilitiesGuardsTypeProtocol": (
        "flext_core._utilities.guards_type_protocol",
        "FlextUtilitiesGuardsTypeProtocol",
    ),
    "FlextUtilitiesMapper": ("flext_core._utilities.mapper", "FlextUtilitiesMapper"),
    "FlextUtilitiesModel": ("flext_core._utilities.model", "FlextUtilitiesModel"),
    "FlextUtilitiesParser": ("flext_core._utilities.parser", "FlextUtilitiesParser"),
    "FlextUtilitiesReliability": (
        "flext_core._utilities.reliability",
        "FlextUtilitiesReliability",
    ),
    "FlextUtilitiesResultHelpers": (
        "flext_core._utilities.result_helpers",
        "FlextUtilitiesResultHelpers",
    ),
    "FlextUtilitiesText": ("flext_core._utilities.text", "FlextUtilitiesText"),
    "args": "flext_core._utilities.args",
    "beartype_conf": "flext_core._utilities.beartype_conf",
    "beartype_engine": "flext_core._utilities.beartype_engine",
    "cache": "flext_core._utilities.cache",
    "checker": "flext_core._utilities.checker",
    "collection": "flext_core._utilities.collection",
    "configuration": "flext_core._utilities.configuration",
    "context": "flext_core._utilities.context",
    "conversion": "flext_core._utilities.conversion",
    "discovery": "flext_core._utilities.discovery",
    "domain": "flext_core._utilities.domain",
    "enforcement": "flext_core._utilities.enforcement",
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

__all__ = [
    "FlextUtilitiesArgs",
    "FlextUtilitiesBeartypeConf",
    "FlextUtilitiesBeartypeEngine",
    "FlextUtilitiesCache",
    "FlextUtilitiesChecker",
    "FlextUtilitiesCollection",
    "FlextUtilitiesConfiguration",
    "FlextUtilitiesContext",
    "FlextUtilitiesConversion",
    "FlextUtilitiesDiscovery",
    "FlextUtilitiesDomain",
    "FlextUtilitiesEnforcement",
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
    "FlextUtilitiesParser",
    "FlextUtilitiesReliability",
    "FlextUtilitiesResultHelpers",
    "FlextUtilitiesText",
    "args",
    "beartype_conf",
    "beartype_engine",
    "cache",
    "checker",
    "collection",
    "configuration",
    "context",
    "conversion",
    "discovery",
    "domain",
    "enforcement",
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
    "parser",
    "reliability",
    "result_helpers",
    "text",
]


install_lazy_exports(__name__, globals(), _LAZY_IMPORTS)
