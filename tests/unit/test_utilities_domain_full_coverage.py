"""Tests for FlextUtilitiesDomain - entity comparison, hashing, validation.

Module: flext_core
Coverage target: lines 32, 156, 197-202, 211

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from datetime import UTC, datetime

from tests import m, t, u


class TestUtilitiesDomainFullCoverage:
    def test_logger_property_returns_logger(self) -> None:
        """Logger property returns a structlog logger (line 32)."""
        domain_util = u()
        logger = domain_util.logger
        assert logger is not None

    def test_hash_with_hashable_non_primitive(self) -> None:
        """Hashable non-primitive value in model_dump is repr'd (line 156)."""

        class EntityWithDate(m.Value):
            unique_id: str = "test"
            created: datetime = datetime(2025, 1, 1, tzinfo=UTC)

        entity = EntityWithDate()
        result = u.hash_value_object_by_value(entity)
        assert isinstance(result, int)

    def test_hash_with_non_hashable_value(self) -> None:
        """Non-hashable value in model_dump uses repr (line 159)."""

        class EntityWithList(m.Value):
            unique_id: str = "test"
            tags: t.StrSequence = m.Field(default_factory=lambda: ["a", "b"])

        entity = EntityWithList(tags=["a", "b"])
        result = u.hash_value_object_by_value(entity)
        assert isinstance(result, int)
