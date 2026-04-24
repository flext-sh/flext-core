"""Constants mixin for domain.

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from enum import StrEnum, unique
from typing import Final

from flext_core import FlextConstants as c


class TestsFlextCoreConstantsDomain:
    class Paths:
        """Path constants for flext-core tests."""

        REPO_ROOT_PARENT_DEPTH: Final[int] = 2
        SRC_DIR: Final[str] = "src"
        CORE_PACKAGE_DIR: Final[str] = "flext_core"
        EXAMPLES_DIR: Final[str] = "examples"
        PYPROJECT_FILENAME: Final[str] = "pyproject.toml"

    class TestDomain:
        """Flext-core-specific domain utilities test constants."""

        ENTITY_NAME_ALICE: Final[str] = "Alice"
        ENTITY_NAME_BOB: Final[str] = "Bob"
        ENTITY_VALUE_10: Final[int] = 10
        ENTITY_VALUE_20: Final[int] = 20
        VALUE_DATA_TEST: Final[str] = "test"
        VALUE_COUNT_5: Final[int] = 5
        VALUE_COUNT_10: Final[int] = 10
        CUSTOM_ID_1: Final[str] = "id1"
        CUSTOM_ID_2: Final[str] = "id2"
        COMPLEX_ITEMS: Final[tuple[str, ...]] = ("a", "b")

    class Domain(c):
        """Domain-specific constants for tests."""

        Status = c.Status
        Currency = c.Currency
        OrderStatus = c.OrderStatus

    class Cqrs(c):
        """CQRS pattern constants for tests."""

        Status = c.Status
        HandlerType = c.HandlerType
        MetricType = c.MetricType
        ProcessingMode = c.ProcessingMode
        ProcessingPhase = c.ProcessingPhase
        BindType = c.BindType
        MergeStrategy = c.MergeStrategy
        HealthStatus = c.HealthStatus
        TokenType = c.TokenType
        SerializationFormat = c.SerializationFormat
        Compression = c.Compression
        Aggregation = c.Aggregation
        Action = c.Action
        PersistenceLevel = c.PersistenceLevel
        TargetFormat = c.TargetFormat
        WarningLevel = c.WarningLevel
        OutputFormat = c.OutputFormat
        Mode = c.Mode
        RegistrationStatus = c.RegistrationStatus

    @unique
    class StatusEnum(StrEnum):
        """Reusable test status enum for test fixtures.

        Standard three-state status enum used across multiple test modules.
        """

        ACTIVE = "active"
        PENDING = "pending"
        INACTIVE = "inactive"

    @unique
    class PriorityEnum(StrEnum):
        """Reusable test priority enum for test fixtures.

        Standard three-level priority enum used across multiple test modules.
        """

        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
