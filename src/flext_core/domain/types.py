"""Domain types compatibility module for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides compatibility imports from shared_types for backward compatibility.
Redirects to shared_types.py to maintain existing imports while keeping clean architecture.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated

from pydantic import ConfigDict
from pydantic import Field

from flext_core.domain.shared_types import BatchSize
from flext_core.domain.shared_types import DatabaseName
from flext_core.domain.shared_types import EntityId
from flext_core.domain.shared_types import EntityStatus
from flext_core.domain.shared_types import EnvironmentLiteral
from flext_core.domain.shared_types import Host
from flext_core.domain.shared_types import JsonDict
from flext_core.domain.shared_types import Password
from flext_core.domain.shared_types import Port
from flext_core.domain.shared_types import TimeoutSeconds
from flext_core.domain.shared_types import UserId
from flext_core.domain.shared_types import Username


# Validation functions
def validate_entity_id(entity_id: str) -> bool:
    """Validate entity ID format."""
    if not entity_id or len(entity_id) > 100:
        return False
    # Allow letters, numbers, hyphens, underscores
    pattern = r"^[a-zA-Z0-9_-]+$"
    return bool(re.match(pattern, entity_id))


def validate_project_name(name: str) -> bool:
    """Validate project name format."""
    if not name or len(name) > 50:
        return False
    # Allow letters, numbers, hyphens, underscores
    pattern = r"^[a-zA-Z][a-zA-Z0-9_-]*$"
    return bool(re.match(pattern, name))


type CreatedAt = Annotated[
    datetime,
    Field(description="Creation timestamp"),
]

type UpdatedAt = Annotated[
    datetime,
    Field(description="Last update timestamp"),
]


__all__ = [
    "BatchSize",
    # Pydantic types
    "ConfigDict",
    # Timestamp types
    "CreatedAt",
    "DatabaseName",
    # Core types
    "EntityId",
    "EntityStatus",
    # Environment
    "EnvironmentLiteral",
    "Host",
    "JsonDict",
    "Password",
    "Port",
    "TimeoutSeconds",
    "UpdatedAt",
    "UserId",
    "Username",
    # Validation functions
    "validate_entity_id",
    "validate_project_name",
    # All shared_types exports
]
