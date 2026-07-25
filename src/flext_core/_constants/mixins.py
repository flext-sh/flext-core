"""FlextConstantsMixins - mixin and handler support constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsMixins:
    """SSOT for mixin/handler decorator support constants."""

    FIELD_ID: Final[str] = "unique_id"
    FIELD_STATUS: Final[str] = "status"
    FIELD_METADATA: Final[str] = "metadata"
    FIELD_ATTRIBUTES: Final[str] = "attributes"
    FIELD_CONTEXT: Final[str] = "context"
    FIELD_HANDLER_MODE: Final[str] = "handler_mode"

    IDENTIFIER_UNKNOWN: Final[str] = "unknown"
    DEFAULT_MAX_WORKERS: Final[int] = 4

    HANDLER_ATTR: Final[str] = "_flext_handler_config_"
    FACTORY_ATTR: Final[str] = "_flext_factory_config_"

    @unique
    class RegistrationScope(StrEnum):
        """Plugin registration scopes for registry operations."""

        INSTANCE = "instance"
        CLASS = "class"

    @unique
    class MethodName(StrEnum):
        """Standard method names used in handler and mixin resolution."""

        HANDLE = "handle"
        PROCESS = "process"
        EXECUTE = "execute"
        PROCESS_COMMAND = "process_command"
        VALIDATE = "validate"
