"""FlextConstantsBase - core primitive constants (SSOT, MRO facade for base scalars).

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import ClassVar


class FlextConstantsBase:
    """SSOT for base primitive constants used across the workspace."""

    NAME: ClassVar[str] = "FLEXT"
    ZERO: ClassVar[int] = 0

    # Published for the module-scope consolidation consumers (fleet-wide,
    # 2026-10-03): the regenerated cli/context surfaces reference this tuple
    # through the constants namespace for isinstance and type-annotation use.
    PRIMITIVES_TYPES: ClassVar[tuple[type[str], type[int], type[float], type[bool]]] = (
        str,
        int,
        float,
        bool,
    )

    # Runtime type-guard tuples: every consumer reaches them through the
    # constants namespace (``c.*``); the typings family never redeclares them.
    NUMERIC_TYPES: ClassVar[tuple[type[int], type[float]]] = (int, float)
    SEQUENCE_PAIR_TYPES: ClassVar[tuple[type, ...]] = (list, tuple)
    STR_BYTES_TYPES: ClassVar[tuple[type[str], type[bytes]]] = (str, bytes)
    STR_BINARY_TYPES: ClassVar[tuple[type[str], type[bytes], type[bytearray]]] = (
        str,
        bytes,
        bytearray,
    )
    SCALAR_TYPES: ClassVar[
        tuple[type[str], type[int], type[float], type[bool], type[datetime]]
    ] = (str, int, float, bool, datetime)
    CONTAINER_TYPES: ClassVar[
        tuple[type[str], type[int], type[float], type[bool], type[datetime], type[Path]]
    ] = (str, int, float, bool, datetime, Path)
    CONTAINER_AND_COLLECTION_TYPES: ClassVar[tuple[type, ...]] = (
        *CONTAINER_TYPES,
        list,
        dict,
        tuple,
    )

    PERCENTAGE_MULTIPLIER: ClassVar[int] = 100
    MILLISECONDS_MULTIPLIER: ClassVar[int] = 1000
    MICROSECONDS_MULTIPLIER: ClassVar[int] = 1000000

    LOCALHOST: ClassVar[str] = "localhost"
    LOOPBACK_IP: ClassVar[str] = "127.0.0.1"
    MIN_PORT: ClassVar[int] = 1
    MAX_PORT: ClassVar[int] = 65535
    MAX_HOSTNAME_LENGTH: ClassVar[int] = 253
    LDAP_PORT: ClassVar[int] = 389
    LDAPS_PORT: ClassVar[int] = 636
    HTTP_PORT: ClassVar[int] = 80
    HTTPS_PORT: ClassVar[int] = 443
    HTTP_ALT_PORT: ClassVar[int] = 8080
    POSTGRES_PORT: ClassVar[int] = 5432
    REDIS_PORT: ClassVar[int] = 6379
    GRPC_PORT: ClassVar[int] = 50051
    ORACLE_PORT: ClassVar[int] = 1521

    HTTP_STATUS_MIN: ClassVar[int] = 100
    HTTP_STATUS_MAX: ClassVar[int] = 599

    DEFAULT_PAGE_SIZE: ClassVar[int] = 10
    MAX_PAGE_SIZE: ClassVar[int] = 1000
    MIN_PAGE_SIZE: ClassVar[int] = 1

    DEFAULT_BACKOFF_MULTIPLIER: ClassVar[float] = 2.0
    DEFAULT_SIZE: ClassVar[int] = 1000
    MS_PER_SECOND: ClassVar[int] = 1000
    MAX_ITEMS: ClassVar[int] = 10000
    DEFAULT_EMPTY_STRING: ClassVar[str] = ""
    DEFAULT_METADATA_SCHEMA_VERSION: ClassVar[str] = "1.0.0"
