# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Protocols package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from types import MappingProxyType
from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._protocols._container.flextprotocolscontainer_part_01 import (
        FlextProtocolsContainerPart01,
    )
    from flext_core._protocols._container.flextprotocolscontainer_part_02 import (
        FlextProtocolsContainerPart02,
    )
    from flext_core._protocols._container.flextprotocolscontainer_part_03 import (
        FlextProtocolsContainerPart03,
    )
    from flext_core._protocols._context.flextprotocolscontext_part_01 import (
        FlextProtocolsContextPart01,
    )
    from flext_core._protocols._context.flextprotocolscontext_part_02 import (
        FlextProtocolsContextPart02,
    )
    from flext_core._protocols._context.flextprotocolscontext_part_03 import (
        FlextProtocolsContextPart03,
    )
    from flext_core._protocols._logging.flextprotocolslogging_part_01 import (
        FlextProtocolsLoggingPart01,
    )
    from flext_core._protocols._logging.flextprotocolslogging_part_02 import (
        FlextProtocolsLoggingPart02,
    )
    from flext_core._protocols._logging.flextprotocolslogging_part_03 import (
        FlextProtocolsLoggingPart03,
    )
    from flext_core._protocols.base import FlextProtocolsBase
    from flext_core._protocols.config import FlextProtocolsConfig
    from flext_core._protocols.container import FlextProtocolsContainer
    from flext_core._protocols.context import FlextProtocolsContext
    from flext_core._protocols.handler import FlextProtocolsHandler
    from flext_core._protocols.loggings import FlextProtocolsLogging
    from flext_core._protocols.project_metadata import FlextProtocolsProjectMetadata
    from flext_core._protocols.pydantic import FlextProtocolsPydantic
    from flext_core._protocols.registry import FlextProtocolsRegistry
    from flext_core._protocols.result import FlextProtocolsResult
    from flext_core._protocols.service import FlextProtocolsService
    from flext_core._protocols.settings import FlextProtocolsSettings


__all__: tuple[str, ...] = (
    "FlextProtocolsBase",
    "FlextProtocolsConfig",
    "FlextProtocolsContainer",
    "FlextProtocolsContainerPart01",
    "FlextProtocolsContainerPart02",
    "FlextProtocolsContainerPart03",
    "FlextProtocolsContext",
    "FlextProtocolsContextPart01",
    "FlextProtocolsContextPart02",
    "FlextProtocolsContextPart03",
    "FlextProtocolsHandler",
    "FlextProtocolsLogging",
    "FlextProtocolsLoggingPart01",
    "FlextProtocolsLoggingPart02",
    "FlextProtocolsLoggingPart03",
    "FlextProtocolsProjectMetadata",
    "FlextProtocolsPydantic",
    "FlextProtocolsRegistry",
    "FlextProtocolsResult",
    "FlextProtocolsService",
    "FlextProtocolsSettings",
)

install_lazy_exports(
    __name__,
    globals(),
    MappingProxyType({
        "FlextProtocolsBase": ".base",
        "FlextProtocolsConfig": ".config",
        "FlextProtocolsContainer": ".container",
        "FlextProtocolsContainerPart01": "._container.flextprotocolscontainer_part_01",
        "FlextProtocolsContainerPart02": "._container.flextprotocolscontainer_part_02",
        "FlextProtocolsContainerPart03": "._container.flextprotocolscontainer_part_03",
        "FlextProtocolsContext": ".context",
        "FlextProtocolsContextPart01": "._context.flextprotocolscontext_part_01",
        "FlextProtocolsContextPart02": "._context.flextprotocolscontext_part_02",
        "FlextProtocolsContextPart03": "._context.flextprotocolscontext_part_03",
        "FlextProtocolsHandler": ".handler",
        "FlextProtocolsLogging": ".loggings",
        "FlextProtocolsLoggingPart01": "._logging.flextprotocolslogging_part_01",
        "FlextProtocolsLoggingPart02": "._logging.flextprotocolslogging_part_02",
        "FlextProtocolsLoggingPart03": "._logging.flextprotocolslogging_part_03",
        "FlextProtocolsProjectMetadata": ".project_metadata",
        "FlextProtocolsPydantic": ".pydantic",
        "FlextProtocolsRegistry": ".registry",
        "FlextProtocolsResult": ".result",
        "FlextProtocolsService": ".service",
        "FlextProtocolsSettings": ".settings",
    }),
    public_exports=__all__,
)
