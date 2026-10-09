"""FlextConstantsMixins - mixin and handler support constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import ClassVar


class FlextConstantsMixins:
    """SSOT for mixin/handler decorator support constants."""

    FIELD_ID: ClassVar[str] = "unique_id"
    FIELD_STATUS: ClassVar[str] = "status"
    FIELD_METADATA: ClassVar[str] = "metadata"
    FIELD_ATTRIBUTES: ClassVar[str] = "attributes"
    FIELD_CONTEXT: ClassVar[str] = "context"
    FIELD_HANDLER_MODE: ClassVar[str] = "handler_mode"

    IDENTIFIER_UNKNOWN: ClassVar[str] = "unknown"
    DEFAULT_MAX_WORKERS: ClassVar[int] = 4

    HANDLER_ATTR: ClassVar[str] = "_flext_handler_config_"
    FACTORY_ATTR: ClassVar[str] = "_flext_factory_config_"

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
