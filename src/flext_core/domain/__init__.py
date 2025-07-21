"""Domain layer - Pure business logic with zero dependencies.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

# Essential domain components only - following KISS principle
from flext_core.domain.core import DomainError
from flext_core.domain.core import Repository
from flext_core.domain.core import ServiceError
from flext_core.domain.core import ValidationError
from flext_core.domain.models import ServiceResult
from flext_core.domain.pydantic_base import DomainAggregateRoot
from flext_core.domain.pydantic_base import DomainEntity
from flext_core.domain.pydantic_base import DomainEvent
from flext_core.domain.pydantic_base import DomainValueObject

# Import other components only when needed
# Use explicit imports in your code for better clarity and reduced coupling

__all__ = [
    # Essential domain abstractions
    "DomainAggregateRoot",
    "DomainEntity",
    # Core functionality
    "DomainError",
    "DomainEvent",
    "DomainValueObject",
    "Repository",
    "ServiceError",
    "ServiceResult",
    "ValidationError",
]
