"""Domain types compatibility module for FLEXT framework.

Copyright (c) 2025 FLEXT Contributors
SPDX-License-Identifier: MIT

This module provides compatibility imports from shared_types for backward compatibility.
Redirects to shared_types.py to maintain existing imports while keeping clean architecture.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

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
    # All shared_types exports
]
