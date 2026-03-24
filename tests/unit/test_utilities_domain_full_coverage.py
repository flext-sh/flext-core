"""Tests for FlextUtilitiesDomain - entity comparison, hashing, validation.

Module: flext_core._utilities.domain
Coverage target: lines 32, 156, 197-202, 211

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from typing import Annotated, cast, override

from pydantic import BaseModel, Field

from flext_core import t, u

from ._models import TestUnitModels


class TestUtilitiesDomainFullCoverage:
    def test_logger_property_returns_logger(self) -> None:
        """Logger property returns a structlog logger (line 32)."""
        domain_util = u()
        logger = domain_util.logger
        assert logger is not None
        assert hasattr(logger, "info")

    def test_hash_with_hashable_non_primitive(self) -> None:
        """Hashable non-primitive value in model_dump is repr'd (line 156)."""

        class EntityWithDate(BaseModel):
            unique_id: str = "test"
            created: datetime = datetime(2025, 1, 1, tzinfo=UTC)

        entity = EntityWithDate()
        result = u.hash_value_object_by_value(entity)
        assert isinstance(result, int)

    def test_hash_with_non_hashable_value(self) -> None:
        """Non-hashable value in model_dump uses repr (line 159)."""

        class EntityWithList(BaseModel):
            unique_id: str = "test"
            tags: Annotated[Sequence[str], Field(default_factory=lambda: ["a", "b"])]

        entity = EntityWithList(tags=["a", "b"])
        result = u.hash_value_object_by_value(entity)
        assert isinstance(result, int)

    def test_frozen_model_is_immutable(self) -> None:
        """Pydantic model with frozen=True detected as immutable (lines 197-200)."""
        entity = TestUnitModels._FrozenEntity()
        assert u.validate_value_object_immutable(entity) is True

    def test_non_frozen_model_checks_setattr(self) -> None:
        """Non-frozen Pydantic model checks __setattr__ override."""
        entity = TestUnitModels._SampleEntity()
        result = u.validate_value_object_immutable(entity)
        assert isinstance(result, bool)

    def test_plain_object_is_mutable(self) -> None:
        """Plain t.NormalizedValue with default __setattr__ is mutable (line 211 branch)."""

        class PlainObj:
            pass

        obj = PlainObj()
        assert (
            u.validate_value_object_immutable(
                cast("t.RuntimeData", obj),
            )
            is False
        )

    def test_object_without_model_config(self) -> None:
        """Object without model_config just checks __setattr__."""
        result = u.validate_value_object_immutable("hello")
        assert isinstance(result, bool)

    def test_validate_value_object_immutable_exception_and_no_setattr_branch(
        self,
    ) -> None:
        class _BrokenConfigDict:
            """Dict-like t.NormalizedValue whose get() raises TypeError."""

            def get(self, key: str, default: bool | None = None) -> bool:
                _ = key
                _ = default
                msg = "bad config"
                raise TypeError(msg)

            def __getitem__(self, key: str) -> bool:
                msg = "bad config"
                raise TypeError(msg)

            def __iter__(self) -> Iterator[str]:
                return iter(())

            def __len__(self) -> int:
                return 0

        class _BrokenConfig:
            model_config: _BrokenConfigDict = _BrokenConfigDict()

        class _NoSetattrVisible:
            @override
            def __getattribute__(self, name: str) -> t.NormalizedValue:
                if name == "__setattr__":
                    raise AttributeError(name)
                return object.__getattribute__(self, name)

        assert (
            u.validate_value_object_immutable(
                cast("t.RuntimeData", _BrokenConfig()),
            )
            is False
        )
        assert (
            u.validate_value_object_immutable(
                cast("t.RuntimeData", _NoSetattrVisible()),
            )
            is False
        )
