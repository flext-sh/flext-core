"""Native state declarations consumed by the operational FlextMixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping
from typing import ClassVar

from typing_extensions import TypeForm

from flext_core import FlextContainer, m, p, t, u
from flext_core._models import FlextContext


class FlextMixins(m.ArbitraryTypesModel):
    """Own service runtime inputs and state without operational behavior."""

    settings_type: t.SkipJsonSchema[t.SettingsClass | None] = m.Field(
        None, validate_default=True, exclude=True,
        description="FlextSettings class used to initialize the service runtime.",
    )
    runtime_settings: t.Port[p.Settings | None] = m.Field(
        None, exclude=True,
        description="Pre-built settings instance used directly for the runtime.",
    )
    settings_overrides: t.SkipJsonSchema[t.ScalarMapping | None] = m.Field(
        None, validate_default=True, exclude=True,
        description="Settings overrides applied at instantiation.",
    )
    initial_context: t.Port[p.Context | None] = m.Field(
        None, exclude=True, description="Initial context for the service scope.",
    )
    _runtime: p.ServiceRuntime | None = u.PrivateAttr(None)
    _operation_stats: MutableMapping[str, m.ConfigMap] = u.PrivateAttr(
        default_factory=u.empty(TypeForm(MutableMapping[str, m.ConfigMap])),
    )
    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}
    _cache_lock: ClassVar[p.Lock] = threading.Lock()
    _container_type: ClassVar[p.ContainerType] = FlextContainer
    _context_type: ClassVar[p.ContextType] = FlextContext
    _auto_context_scope: ClassVar[bool] = True


__all__: t.StrSequence = ("FlextMixins",)
