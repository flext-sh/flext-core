"""FLEXT Settings mixins — MRO composition layer.

Each submodule exposes one concern (singleton, core fields, database, etc.)
so ``FlextSettings`` can compose them via plain multiple inheritance.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from flext_core._settings.context import FlextSettingsContext
from flext_core._settings.core import FlextSettingsCore
from flext_core._settings.database import FlextSettingsDatabase
from flext_core._settings.di import FlextSettingsDI
from flext_core._settings.dispatcher import FlextSettingsDispatcher
from flext_core._settings.infrastructure import FlextSettingsInfrastructure
from flext_core._settings.registry import FlextSettingsRegistry

__all__: list[str] = [
    "FlextSettingsContext",
    "FlextSettingsCore",
    "FlextSettingsDI",
    "FlextSettingsDatabase",
    "FlextSettingsDispatcher",
    "FlextSettingsInfrastructure",
    "FlextSettingsRegistry",
]
