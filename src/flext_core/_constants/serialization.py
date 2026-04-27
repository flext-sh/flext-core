"""FlextConstantsSerialization - serialization and encoding constants (SSOT).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final


class FlextConstantsSerialization:
    """SSOT for serialization format, compression, and encoding constants."""

    @unique
    class SerializationFormat(StrEnum):
        """Serialization format identifiers."""

        JSON = "json"
        YAML = "yaml"
        TOML = "toml"
        MSGPACK = "msgpack"

    @unique
    class Compression(StrEnum):
        """Compression algorithm identifiers."""

        NONE = "none"
        GZIP = "gzip"
        BZIP2 = "bzip2"
        LZ4 = "lz4"

    @unique
    class DecodeErrorHandler(StrEnum):
        """Python bytes.decode error-handler identifiers (SSOT for ``errors=`` param)."""

        REPLACE = "replace"
        IGNORE = "ignore"
        STRICT = "strict"
        BACKSLASHREPLACE = "backslashreplace"
        XMLCHARREFREPLACE = "xmlcharrefreplace"

    DEFAULT_ENCODING: Final[str] = "utf-8"
    DEFAULT_DECODE_ERROR_HANDLER: Final[str] = DecodeErrorHandler.REPLACE
