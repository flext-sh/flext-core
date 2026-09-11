"""Runtime settings for flext-infra — namespaced under ``settings.Infra``.

Layer-0 style: universal fields via FLEXT; all project fields in the ``Infra``
namespace group with simple scalar types (env-settable).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import ClassVar

from pydantic_settings import SettingsConfigDict

from flext_core import FlextSettings
from flext_infra import m
from flext_infra._models.settings import FlextInfraSettingsModels


# NOTE (multi-agent): migrated base FlextCliSettings->FlextSettings to
# complete the workspace settings migration (flext-d421); flext_cli dropped its
# public FlextCliSettings export. Canonical pattern per flext-api/flext-auth.
class _FlextInfraSettings(FlextSettings):
    """Environment-backed infra settings; fields under ``settings.Infra.*``."""

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        env_prefix="FLEXT_INFRA_",
        env_nested_delimiter="__",
        extra="ignore",
        frozen=True,
    )

    # flext-wkii.4.15: composition only; declaration and env validation stay private.
    Infra: FlextInfraSettingsModels.Infra = m.Field(
        default_factory=FlextInfraSettingsModels.Infra,
        description="Namespaced infra settings.",
    )


settings: _FlextInfraSettings = _FlextInfraSettings()
"""Pre-instantiated project settings singleton — ``from flext_infra import settings``."""

__all__: list[str] = ["settings"]
