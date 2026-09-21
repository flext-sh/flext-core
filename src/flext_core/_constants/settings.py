"""FlextConstantsSettings - runtime settings constants.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Final, Self

from pydantic import ConfigDict

from .._settings import FlextSettings


class FlextConstantsSettings(FlextSettings):
    """SSOT for runtime settings.

    MRO carries ``FlextSettings`` (ENFORCE-042); the class is a namespace
    holder, never instantiated — class-attribute access resolves via the MRO.
    """

    # ENFORCE-042 namespace-holder contract: ``FlextSettings`` contributes
    # namespacing only — instance machinery stays plain object semantics so the
    # settings singleton/validation machinery cannot leak into instantiated
    # facade composites (e.g. the ``u`` logging facade).
    def __new__(cls, *args: object, **kwargs: object) -> Self:
        return object.__new__(cls)

    def __init__(self, *args: object, **kwargs: object) -> None:
        _ = self, args, kwargs

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    __eq__ = object.__eq__

    __hash__ = object.__hash__

    SHORT_UUID_LENGTH: Final[int] = 8

    ENV_FILE_ENV_VAR: Final[str] = "FLEXT_ENV_FILE"
    """Bootstrap env var that overrides the .env path (settings protocol owner)."""

    EXTRA_CONFIG_FORBID: Final = "forbid"
    EXTRA_CONFIG_IGNORE: Final = "ignore"
    SERIALIZATION_ISO8601: Final = "iso8601"
    SERIALIZATION_BASE64: Final = "base64"

    DOMAIN_MODEL_CONFIG: Final[ConfigDict] = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        validate_return=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=False,
        extra="forbid",
    )
    """Domain model configuration defaults (SSOT; consumed via ``c.*``)."""
