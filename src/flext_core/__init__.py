"""FLext Core - Enterprise Foundation Framework.

Copyright (c) 2024 FLEXT Contributors
SPDX-License-Identifier: MIT

Modern Python 3.13 + Pydantic v2 + Clean Architecture.
Zero tolerance for code duplication and technical debt.

Key Principles:
- SOLID: Single responsibility, Open/closed, Liskov substitution,
    Interface segregation, Dependency inversion
- KISS: Keep it simple, stupid
- DRY: Don't repeat yourself
- Performance: Zero overhead abstractions
- Type Safety: 100% typed with modern Python 3.13 syntax
"""

from __future__ import annotations

# Essential external dependencies
from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings as PydanticBaseSettings

# Version information
from flext_core.domain.constants import FlextFramework

__version__ = FlextFramework.VERSION

# Core domain patterns - most commonly used across FLEXT
# Essential configuration - dependency injection and settings
from flext_core.config.base import BaseConfig
from flext_core.config.base import BaseSettings
from flext_core.config.base import DIContainer
from flext_core.config.base import configure_container
from flext_core.config.base import get_container
from flext_core.config.base import injectable
from flext_core.config.base import singleton
from flext_core.domain.core import DomainError
from flext_core.domain.core import ValidationError
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainBaseModel
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject
from flext_core.domain.shared_models import LogLevel
from flext_core.domain.shared_types import EntityId
from flext_core.domain.shared_types import UserId
from flext_core.domain.types import ServiceResult

# Clean Architecture Public API - Essential patterns only
# For specific functionality, import from sub-modules:
# - flext_core.config.* for configuration
# - flext_core.domain.* for domain models
# - flext_core.infrastructure.* for infrastructure
__all__ = [
    # Essential configuration (dependency injection)
    "BaseConfig",
    # Essential Pydantic exports
    "BaseModel",
    "BaseSettings",
    "DIContainer",
    "DomainAggregateRoot",
    "DomainBaseModel",
    "DomainEntity",
    "DomainError",
    "DomainEvent",
    "DomainValueObject",
    "EntityId",
    "Field",
    "FlextFramework",
    "LogLevel",
    "PydanticBaseSettings",
    # Core domain patterns (most used)
    "ServiceResult",
    "UserId",
    "ValidationError",
    # Version
    "__version__",
    "configure_container",
    "get_container",
    "injectable",
    "singleton",
]
