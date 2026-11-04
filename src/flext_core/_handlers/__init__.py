# AUTO-GENERATED FILE — Regenerate with: make gen
"""Flext Core. Handlers package.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_core.lazy import install_lazy_exports

if TYPE_CHECKING:
    from flext_core._handlers.flexthandlers_cqrs_base import FlextHandlersCqrsBase
    from flext_core._handlers.flexthandlers_discovery import FlextHandlersDiscovery
    from flext_core._handlers.flexthandlers_dispatch import FlextHandlersDispatch
    from flext_core._handlers.flexthandlers_execution import FlextHandlersExecution
    from flext_core._handlers.flexthandlers_factory import FlextHandlersFactory
    from flext_core._handlers.flexthandlers_pipeline import FlextHandlersPipeline
    from flext_core._handlers.flexthandlers_settings import FlextHandlersSettings


__all__: tuple[str, ...] = (
    "FlextHandlersCqrsBase",
    "FlextHandlersDiscovery",
    "FlextHandlersDispatch",
    "FlextHandlersExecution",
    "FlextHandlersFactory",
    "FlextHandlersPipeline",
    "FlextHandlersSettings",
)

install_lazy_exports(
    __name__,
    globals(),
    {
        "FlextHandlersCqrsBase": ".flexthandlers_cqrs_base",
        "FlextHandlersDiscovery": ".flexthandlers_discovery",
        "FlextHandlersDispatch": ".flexthandlers_dispatch",
        "FlextHandlersExecution": ".flexthandlers_execution",
        "FlextHandlersFactory": ".flexthandlers_factory",
        "FlextHandlersPipeline": ".flexthandlers_pipeline",
        "FlextHandlersSettings": ".flexthandlers_settings",
    },
    public_exports=__all__,
)
