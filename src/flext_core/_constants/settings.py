"""FlextConstantsSettings - runtime settings constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar, Literal

from pydantic import ConfigDict

from flext_core._settings import FlextSettings


class FlextConstantsSettings:
    """SSOT for runtime settings."""

    SHORT_UUID_LENGTH: ClassVar[int] = 8

    ENV_FILE_ENV_VAR: ClassVar[str] = FlextSettings.ENV_FILE_ENV_VAR
    """Bootstrap env var that overrides the .env path (settings protocol owner)."""

    ENV_FILE_DEFAULT: ClassVar[str] = FlextSettings.ENV_FILE_DEFAULT

    EXTRA_CONFIG_FORBID: ClassVar[Literal["forbid"]] = "forbid"
    EXTRA_CONFIG_IGNORE: ClassVar[Literal["ignore"]] = "ignore"
    SERIALIZATION_ISO8601: ClassVar[Literal["iso8601"]] = "iso8601"
    SERIALIZATION_BASE64: ClassVar[Literal["base64"]] = "base64"

    MODEL_USE_ENUM_VALUES: ClassVar[bool] = False
    MODEL_VALIDATE_DEFAULT: ClassVar[bool] = True
    MODEL_DEFER_BUILD: ClassVar[bool] = True

    DOMAIN_MODEL_CONFIG: ClassVar = MappingProxyType(
        ConfigDict(
            use_enum_values=MODEL_USE_ENUM_VALUES,
            validate_assignment=True,
            validate_return=True,
            validate_default=MODEL_VALIDATE_DEFAULT,
            str_strip_whitespace=True,
            arbitrary_types_allowed=False,
            extra="forbid",
        ),
    )
    """Domain model configuration defaults (SSOT; consumed via ``c.*``)."""


__all__: tuple[str, ...] = ("FlextConstantsSettings",)
