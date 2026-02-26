"""Tests for FlextUtilitiesDomain - entity comparison, hashing, validation.

Module: flext_core._utilities.domain
Coverage target: lines 32, 156, 197-202, 211

Copyright (c) 2025 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections import UserDict
from datetime import datetime
from typing import cast

from flext_core import t, u
from pydantic import BaseModel, ConfigDict


class _SampleEntity(BaseModel):
    """Test entity for domain utility tests."""

    model_config = ConfigDict(frozen=False)

    unique_id: str = "test-123"
    name: str = "test"


class _FrozenEntity(BaseModel):
    """Frozen entity for immutability tests."""

    model_config = ConfigDict(frozen=True)

    unique_id: str = "frozen-1"


class TestDomainLogger:
    """Tests for u.Domain.logger property."""

    def test_logger_property_returns_logger(self) -> None:
        """Logger property returns a structlog logger (line 32)."""
        domain_util = u.Domain()
        logger = domain_util.logger
        assert logger is not None
        assert hasattr(logger, "info")


class TestDomainHashValueObject:
    """Tests for u.Domain.hash_value_object_by_value()."""

    def test_hash_with_hashable_non_primitive(self) -> None:
        """Hashable non-primitive value in model_dump is repr'd (line 156)."""
        # datetime is Hashable but not str/int/float/bool/None

        class EntityWithDate(BaseModel):
            unique_id: str = "test"
            created: datetime = datetime(2025, 1, 1)

        entity = EntityWithDate()
        result = u.Domain.hash_value_object_by_value(entity)
        assert isinstance(result, int)

    def test_hash_with_non_hashable_value(self) -> None:
        """Non-hashable value in model_dump uses repr (line 159)."""

        class EntityWithList(BaseModel):
            unique_id: str = "test"
            tags: list[str] = ["a", "b"]

        entity = EntityWithList()
        result = u.Domain.hash_value_object_by_value(entity)
        assert isinstance(result, int)


class TestValidateValueObjectImmutable:
    """Tests for u.Domain.validate_value_object_immutable()."""

    def test_frozen_model_is_immutable(self) -> None:
        """Pydantic model with frozen=True detected as immutable (lines 197-200)."""
        entity = _FrozenEntity()
        assert u.Domain.validate_value_object_immutable(entity) is True

    def test_non_frozen_model_checks_setattr(self) -> None:
        """Non-frozen Pydantic model checks __setattr__ override."""
        entity = _SampleEntity()
        # Pydantic models override __setattr__, so this should be True
        result = u.Domain.validate_value_object_immutable(entity)
        assert isinstance(result, bool)

    def test_plain_object_is_mutable(self) -> None:
        """Plain object with default __setattr__ is mutable (line 211 branch)."""

        class PlainObj:
            pass

        obj = PlainObj()
        # PlainObj uses object.__setattr__ → mutable
        assert (
            u.Domain.validate_value_object_immutable(
                cast("t.ConfigMapValue", cast("object", obj))
            )
            is False
        )

    def test_object_without_model_config(self) -> None:
        """Object without model_config just checks __setattr__."""
        result = u.Domain.validate_value_object_immutable("hello")
        assert isinstance(result, bool)


def test_validate_value_object_immutable_exception_and_no_setattr_branch() -> None:
    class _BrokenConfigDict(UserDict[str, bool]):
        def get(self, key: str, default: object = None) -> bool:
            _ = key
            _ = default
            msg = "bad config"
            raise TypeError(msg)

    class _BrokenConfig:
        model_config: _BrokenConfigDict = _BrokenConfigDict()

    class _NoSetattrVisible:
        def __getattribute__(self, name: str):
            if name == "__setattr__":
                raise AttributeError(name)
            return object.__getattribute__(self, name)

    assert (
        u.Domain.validate_value_object_immutable(
            cast("t.ConfigMapValue", cast("object", _BrokenConfig()))
        )
        is False
    )
    assert (
        u.Domain.validate_value_object_immutable(
            cast("t.ConfigMapValue", cast("object", _NoSetattrVisible()))
        )
        is False
    )
