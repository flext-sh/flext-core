"""Foundation Layer - Absolute Core Abstractions.

This module provides the most fundamental abstractions and patterns
that everything else in FLEXT is built upon.

🏗️ FOUNDATION PRINCIPLE:
This layer contains ONLY pure abstractions, interfaces, and base patterns.
NO concrete implementations, NO external dependencies (except Python stdlib).
"""

from __future__ import annotations

from flext_core.foundation.abstractions import AbstractEntity
from flext_core.foundation.abstractions import AbstractRepository
from flext_core.foundation.abstractions import AbstractService
from flext_core.foundation.abstractions import AbstractValueObject
from flext_core.foundation.patterns import ResultPattern
from flext_core.foundation.patterns import SpecificationPattern
from flext_core.foundation.primitives import EntityId
from flext_core.foundation.primitives import Timestamp
from flext_core.foundation.primitives import UserId
from flext_core.foundation.protocols import EventBus
from flext_core.foundation.protocols import Serializable
from flext_core.foundation.protocols import Validatable

__all__ = [
    # Core abstractions
    "AbstractEntity",
    "AbstractRepository",
    "AbstractService",
    "AbstractValueObject",
    # Primitive types
    "EntityId",
    # Protocols
    "EventBus",
    # Architectural patterns
    "ResultPattern",
    "Serializable",
    "SpecificationPattern",
    "Timestamp",
    "UserId",
    "Validatable",
]
