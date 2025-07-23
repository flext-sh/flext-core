"""FLEXT Core Domain - Domain-Driven Design Building Blocks.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

Domain-Driven Design building blocks following software engineering
principles. Provides foundation classes for implementing domain models
with type safety, immutability, and business rule validation using
Pydantic V2.

Module structure following SOLID principles:
- entity.py: FlextEntity base class for domain entities
- value_object.py: FlextValueObject base class for value objects
- aggregate_root.py: FlextAggregateRoot base class for aggregate roots
- domain_service.py: FlextDomainService base class for domain services

All classes maintain quality with validation, thread safety,
and production patterns.
"""

from __future__ import annotations

# Domain-Driven Design Building Blocks
from .aggregate_root import FlextAggregateRoot
from .domain_service import FlextDomainService
from .entity import FlextEntity
from .value_object import FlextValueObject

# Public API exports
__all__ = [
    "FlextAggregateRoot",
    "FlextDomainService",
    "FlextEntity",
    "FlextValueObject",
]
