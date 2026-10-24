"""Native state declarations consumed by the operational FlextMixins facade.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import threading
from collections.abc import MutableMapping
from typing import ClassVar

from flext_core import p, t
from flext_core._models import FlextModelsBase, FlextModelsPydantic


class FlextMixins(FlextModelsBase.ArbitraryTypesModel):
    """Own service runtime inputs and state without operational behavior."""

    settings_type: t.SkipJsonSchema[p.SettingsType | None] = FlextModelsPydantic.Field(
        default=None,
        validate_default=True,
        exclude=True,
        description="FlextSettings class used to initialize the service runtime.",
    )
    runtime_settings: t.Port[p.Settings] | None = FlextModelsPydantic.Field(
        default=None,
        exclude=True,
        description="Pre-built settings instance used directly for the runtime.",
    )
    settings_overrides: t.SkipJsonSchema[t.JsonMapping | None] = (
        FlextModelsPydantic.Field(
            default=None,
            validate_default=True,
            exclude=True,
            description="Settings overrides applied at instantiation.",
        )
    )
    initial_context: t.Port[p.Context] | None = FlextModelsPydantic.Field(
        default=None,
        exclude=True,
        description="Initial context for the service scope.",
    )
    _runtime: p.ServiceRuntime | None = FlextModelsPydantic.PrivateAttr(None)
    _operation_stats: MutableMapping[str, p.MutableRootDict[t.JsonPayload]] = (
        FlextModelsPydantic.PrivateAttr(
            default_factory=dict[str, p.MutableRootDict[t.JsonPayload]],
        )
    )
    _logger_cache: ClassVar[MutableMapping[str, p.Logger]] = {}
    _cache_lock: ClassVar[p.Lock] = threading.Lock()
    _auto_context_scope: ClassVar[bool] = True


__all__: t.StrSequence = ("FlextMixins",)
