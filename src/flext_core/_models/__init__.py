# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Models package."""

from __future__ import annotations

from typing import TYPE_CHECKING

# mro-i6nq.10: The package consumes its manifest's public-export contract.
from flext_core._models.__unit__ import (
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
    from flext_core._models import (
        _base_parts as _base_parts,
        _container_parts as _container_parts,
        _context as _context,
        _enforcement as _enforcement,
        _exception_params_parts as _exception_params_parts,
    )
    from flext_core._models.base import FlextModelsBase as FlextModelsBase
    from flext_core._models.builder import FlextModelsBuilder as FlextModelsBuilder
    from flext_core._models.collections import (
        FlextModelsCollections as FlextModelsCollections,
    )
    from flext_core._models.config import FlextModelsConfig as FlextModelsConfig
    from flext_core._models.container import (
        FlextModelsContainer as FlextModelsContainer,
    )
    from flext_core._models.containers import (
        FlextModelsContainers as FlextModelsContainers,
        mc as mc,
    )
    from flext_core._models.context import FlextModelsContext as FlextModelsContext
    from flext_core._models.cqrs import FlextModelsCqrs as FlextModelsCqrs
    from flext_core._models.dispatcher import (
        FlextModelsDispatcher as FlextModelsDispatcher,
    )
    from flext_core._models.domain_event import (
        FlextModelsDomainEvent as FlextModelsDomainEvent,
    )
    from flext_core._models.enforcement import (
        FlextModelsEnforcement as FlextModelsEnforcement,
    )
    from flext_core._models.entity import FlextModelsEntity as FlextModelsEntity
    from flext_core._models.errors import FlextModelsErrors as FlextModelsErrors
    from flext_core._models.exception_params import (
        FlextModelsExceptionParams as FlextModelsExceptionParams,
    )
    from flext_core._models.handler import FlextModelsHandler as FlextModelsHandler
    from flext_core._models.namespace import (
        FlextModelsNamespace as FlextModelsNamespace,
    )
    from flext_core._models.project_metadata import (
        FlextModelsProjectMetadata as FlextModelsProjectMetadata,
    )
    from flext_core._models.pydantic import FlextModelsPydantic as FlextModelsPydantic
    from flext_core._models.registry import FlextModelsRegistry as FlextModelsRegistry
    from flext_core._models.service import FlextModelsService as FlextModelsService
    from flext_core._models.settings import FlextModelsSettings as FlextModelsSettings

    # mro-i6nq.10: Static declaration mirrors the installer-owned runtime binding.
    __all__: tuple[str, ...]


_LAZY_IMPORTS = merge_lazy_imports(
    _CHILD_MODULE_PATHS,
    build_lazy_import_map(
        _LAZY_MODULES, alias_groups=_LAZY_ALIAS_GROUPS, sort_keys=False
    ),
    exclude_names=_EXCLUDED_LAZY_NAMES,
    module_name=__name__,
)


# mro-i6nq.10: The installer publishes __all__ from the manifest's literal ABI.
install_lazy_exports(__name__, globals(), _LAZY_IMPORTS, public_exports=_PUBLIC_EXPORTS)
