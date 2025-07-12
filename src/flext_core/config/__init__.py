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
from flext_core.config.validators import validate_database_url
from flext_core.config.validators import validate_port
from flext_core.config.validators import validate_timeout
from flext_core.config.validators import validate_url

__all__ = [
    # Base classes
    "BaseConfig",
    "BaseSettings",
    "ConfigSection",
    "ConfigurationError",
    # Dependency Injection
    "DIContainer",
    "configure_container",
    "get_config",
    "get_container",
    "get_settings",
    "injectable",
    "singleton",
    # Validators
    "validate_database_url",
    "validate_port",
    "validate_timeout",
    "validate_url",
]
