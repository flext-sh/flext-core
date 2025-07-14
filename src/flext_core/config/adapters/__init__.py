"""Configuration adapters package.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Framework-specific adapters for FLEXT configuration system.
"""

from __future__ import annotations

from flext_core.config.adapters.cli import CLIConfig
from flext_core.config.adapters.cli import CLISettings
from flext_core.config.adapters.cli import cli_config_to_dict
from flext_core.config.adapters.django import DjangoSettings
from flext_core.config.adapters.django import django_settings_adapter
from flext_core.config.adapters.singer import SingerConfig
from flext_core.config.adapters.singer import SingerSettings
from flext_core.config.adapters.singer import SingerTapConfig
from flext_core.config.adapters.singer import SingerTargetConfig
from flext_core.config.adapters.singer import singer_config_adapter

__all__ = [
    # CLI
    "CLIConfig",
    "CLISettings",
    # Django
    "DjangoSettings",
    # Singer
    "SingerConfig",
    "SingerSettings",
    "SingerTapConfig",
    "SingerTargetConfig",
    "cli_config_to_dict",
    "django_settings_adapter",
    "singer_config_adapter",
]
