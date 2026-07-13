# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Utilities package."""

from __future__ import annotations

from typing import TYPE_CHECKING

# mro-i6nq.10: The package consumes its manifest's public-export contract.
from flext_core._utilities.__unit__ import (
    CHILD_MODULE_PATHS as _CHILD_MODULE_PATHS,
    EXCLUDED_LAZY_NAMES as _EXCLUDED_LAZY_NAMES,
    LAZY_ALIAS_GROUPS as _LAZY_ALIAS_GROUPS,
    LAZY_MODULES as _LAZY_MODULES,
    PUBLIC_EXPORTS as _PUBLIC_EXPORTS,
)
from flext_core.lazy import (
    build_lazy_import_map,
    install_lazy_exports,
    merge_lazy_imports,
)

if TYPE_CHECKING:
    from flext_core._utilities import (
        _beartype as _beartype,
        _checker_parts as _checker_parts,
        _enforcement_collect_parts as _enforcement_collect_parts,
        _enforcement_parts as _enforcement_parts,
        _logging_config_parts as _logging_config_parts,
        _logging_context_parts as _logging_context_parts,
        _mapper_access_parts as _mapper_access_parts,
        _mapper_extract_parts as _mapper_extract_parts,
        _parser_targets_parts as _parser_targets_parts,
    )
    from flext_core._utilities._context_crud_set import (
        FlextUtilitiesContextCrudSetMixin as FlextUtilitiesContextCrudSetMixin,
    )
    from flext_core._utilities._guards_type_protocol_specs import (
        FlextUtilitiesGuardsTypeProtocolSpecsMixin as FlextUtilitiesGuardsTypeProtocolSpecsMixin,
    )
    from flext_core._utilities._guards_type_protocol_string import (
        FlextUtilitiesGuardsTypeProtocolStringMixin as FlextUtilitiesGuardsTypeProtocolStringMixin,
    )
    from flext_core._utilities._guards_type_protocol_types import (
        ProtocolGuardInput as ProtocolGuardInput,
    )
    from flext_core._utilities.args import FlextUtilitiesArgs as FlextUtilitiesArgs
    from flext_core._utilities.beartype_conf import (
        FlextUtilitiesBeartypeConf as FlextUtilitiesBeartypeConf,
    )
    from flext_core._utilities.beartype_engine import (
        FlextUtilitiesBeartypeEngine as FlextUtilitiesBeartypeEngine,
        ube as ube,
    )
    from flext_core._utilities.beartype_typingext_patch import (
        FlextUtilitiesBeartypeTypingExtPatch as FlextUtilitiesBeartypeTypingExtPatch,
    )
    from flext_core._utilities.checker import (
        FlextUtilitiesChecker as FlextUtilitiesChecker,
    )
    from flext_core._utilities.collection import (
        FlextUtilitiesCollection as FlextUtilitiesCollection,
    )
    from flext_core._utilities.collection_iter import (
        FlextUtilitiesCollectionIter as FlextUtilitiesCollectionIter,
    )
    from flext_core._utilities.collection_merge import (
        FlextUtilitiesCollectionMerge as FlextUtilitiesCollectionMerge,
    )
    from flext_core._utilities.config import (
        FlextUtilitiesConfig as FlextUtilitiesConfig,
    )
    from flext_core._utilities.context import (
        FlextUtilitiesContext as FlextUtilitiesContext,
    )
    from flext_core._utilities.context_crud import (
        FlextUtilitiesContextCrud as FlextUtilitiesContextCrud,
    )
    from flext_core._utilities.context_lifecycle import (
        FlextUtilitiesContextLifecycle as FlextUtilitiesContextLifecycle,
    )
    from flext_core._utilities.context_state import (
        FlextUtilitiesContextState as FlextUtilitiesContextState,
    )
    from flext_core._utilities.conversion import (
        FlextUtilitiesConversion as FlextUtilitiesConversion,
    )
    from flext_core._utilities.discovery import (
        FlextUtilitiesDiscovery as FlextUtilitiesDiscovery,
    )
    from flext_core._utilities.dispatcher_execute import (
        execute_dispatcher_handler as execute_dispatcher_handler,
    )
    from flext_core._utilities.domain import (
        FlextUtilitiesDomain as FlextUtilitiesDomain,
    )
    from flext_core._utilities.enforcement import (
        PREDICATE_BINDINGS as PREDICATE_BINDINGS,
        FlextUtilitiesEnforcement as FlextUtilitiesEnforcement,
    )
    from flext_core._utilities.enforcement_collect import (
        FlextUtilitiesEnforcementCollect as FlextUtilitiesEnforcementCollect,
    )
    from flext_core._utilities.enforcement_emit import (
        FlextUtilitiesEnforcementEmit as FlextUtilitiesEnforcementEmit,
    )
    from flext_core._utilities.enum import FlextUtilitiesEnum as FlextUtilitiesEnum
    from flext_core._utilities.generators import (
        FlextUtilitiesGenerators as FlextUtilitiesGenerators,
    )
    from flext_core._utilities.guards import (
        FlextUtilitiesGuards as FlextUtilitiesGuards,
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
    from flext_core._utilities.handler import (
        FlextUtilitiesHandler as FlextUtilitiesHandler,
    )
    from flext_core._utilities.logging_config import (
        FlextUtilitiesLoggingConfig as FlextUtilitiesLoggingConfig,
    )
    from flext_core._utilities.logging_context import (
        FlextUtilitiesLoggingContext as FlextUtilitiesLoggingContext,
    )
    from flext_core._utilities.mapper import (
        FlextUtilitiesMapper as FlextUtilitiesMapper,
    )
    from flext_core._utilities.mapper_access import (
        FlextUtilitiesMapperAccess as FlextUtilitiesMapperAccess,
    )
    from flext_core._utilities.mapper_extract import (
        FlextUtilitiesMapperExtract as FlextUtilitiesMapperExtract,
    )
    from flext_core._utilities.model import FlextUtilitiesModel as FlextUtilitiesModel
    from flext_core._utilities.model_options import (
        FlextUtilitiesModelOptions as FlextUtilitiesModelOptions,
    )
    from flext_core._utilities.model_runtime import (
        FlextUtilitiesModelRuntime as FlextUtilitiesModelRuntime,
    )
    from flext_core._utilities.parser import (
        FlextUtilitiesParser as FlextUtilitiesParser,
    )
    from flext_core._utilities.parser_coerce import (
        FlextUtilitiesParserCoerce as FlextUtilitiesParserCoerce,
    )
    from flext_core._utilities.parser_targets import (
        FlextUtilitiesParserTargets as FlextUtilitiesParserTargets,
    )
    from flext_core._utilities.project_metadata import (
        FlextUtilitiesProjectMetadata as FlextUtilitiesProjectMetadata,
    )
    from flext_core._utilities.pydantic import (
        FlextUtilitiesPydantic as FlextUtilitiesPydantic,
    )
    from flext_core._utilities.reliability import (
        FlextUtilitiesReliability as FlextUtilitiesReliability,
    )
    from flext_core._utilities.runtime_violation_registry import (
        FlextUtilitiesRuntimeViolationRegistry as FlextUtilitiesRuntimeViolationRegistry,
    )
    from flext_core._utilities.settings import (
        FlextUtilitiesSettings as FlextUtilitiesSettings,
    )
    from flext_core._utilities.text import FlextUtilitiesText as FlextUtilitiesText

    # mro-i6nq.10: Static declaration mirrors the installer-owned runtime binding.
    __all__: tuple[str, ...]


_LAZY_IMPORTS = merge_lazy_imports(
    _CHILD_MODULE_PATHS,
    build_lazy_import_map(
        _LAZY_MODULES,
        alias_groups=_LAZY_ALIAS_GROUPS,
        sort_keys=False,
    ),
    exclude_names=_EXCLUDED_LAZY_NAMES,
    module_name=__name__,
)


# mro-i6nq.10: The installer publishes __all__ from the manifest's literal ABI.
install_lazy_exports(
    __name__,
    globals(),
    _LAZY_IMPORTS,
    public_exports=_PUBLIC_EXPORTS,
)
