"""Enhanced FLEXT Configuration System.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides a unified configuration system for all FLEXT projects,
built on top of Pydantic and pydantic-settings.

Key features:
- Type-safe configuration with Pydantic models
- Environment variable support via pydantic-settings
- Dependency injection container for clean architecture
- Declarative configuration patterns
- Framework-specific adapters (Singer, Django, CLI)
- Validation and serialization
- Configuration inheritance and composition
"""

from __future__ import annotations

from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings
from flext_core.config.base import ConfigSection
from flext_core.config.base import ConfigurationError
from flext_core.config.base import DIContainer
from flext_core.config.base import configure_container
from flext_core.config.base import get_config
from flext_core.config.base import get_container
from flext_core.config.base import get_settings
from flext_core.config.base import injectable
from flext_core.config.base import singleton
from flext_core.config.patterns import BaseFlextSettings

# Import unified config mixins - NO FALLBACKS, usar implementações originais
from flext_core.config.unified_config import AuthConfigMixin
from flext_core.config.unified_config import BaseConfigMixin
from flext_core.config.unified_config import DatabaseConfigMixin
from flext_core.config.unified_config import LoggingConfigMixin
from flext_core.config.unified_config import MonitoringConfigMixin
from flext_core.config.unified_config import PerformanceConfigMixin
from flext_core.config.unified_config import RedisConfigMixin

__all__ = [
    # Unified Config Mixins (conditionally imported)
    "AuthConfigMixin",
    # Base classes
    "BaseConfig",
    "BaseConfigMixin",
    "BaseFlextSettings",
    "BaseSettings",
    "ConfigSection",
    "ConfigurationError",
    # Dependency Injection
    "DIContainer",
    "DatabaseConfigMixin",
    "LoggingConfigMixin",
    "MonitoringConfigMixin",
    "PerformanceConfigMixin",
    "RedisConfigMixin",
    "configure_container",
    "get_config",
    "get_container",
    "get_settings",
    "injectable",
    "singleton",
]
