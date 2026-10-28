"""Behavioral tests for base model mixins and value models.

Asserts the observable public contract of the FlextModelsBase family
(Metadata, ContractModel, FrozenValueModel, IdentifiableMixin,
TimestampableMixin, TimestampedModel) through the public API only:
constructed model state, model_dump output, and validation error paths.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from collections.abc import (
    Iterable,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)

from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Annotated

import pytest

from tests import c, m, t, tm, u


def _is_same(expected: object) -> Callable[[object], bool]:
    """Build a typed identity predicate for ``tm.that(where=...)``.

    Returns:
        The resulting ``Callable[[object], bool]``.

    """

    def matches(candidate: object) -> bool:
        return candidate is expected

    return matches


class TestsFlextCoreModelsBaseFullCoverage:
    """Tests for ``FlextModelsCoreBaseFullCoverage``."""

    class _FrozenValue(m.FrozenValueModel):
        name: Annotated[str, m.Field(description="Frozen value name")]
        count: Annotated[int, m.Field(description="Frozen value count")]

        def __hash__(self) -> int:
            hash_value: int = m.FrozenValueModel.__hash__(self)
            return hash_value

    class _Identifiable(m.FlexibleModel, m.IdentifiableMixin):
        pass

    class _Timestampable(m.FlexibleModel, m.TimestampableMixin):
        pass

    # --- Metadata.attributes coercion contract ---------------------------

    @staticmethod
    def test_metadata_attributes_none_coerces_to_empty_mapping() -> None:
        """Test metadata attributes none coerces to empty mapping."""
        model = m.Metadata.model_validate({"attributes": None})
        tm.that(model.attributes, eq={})

    @staticmethod
    def test_metadata_attributes_accepts_plain_mapping() -> None:
        """Test metadata attributes accepts plain mapping."""
        model = m.Metadata.model_validate({"attributes": {"key": "value"}})
        tm.that(model.attributes, eq={"key": "value"})

    @staticmethod
    def test_metadata_attributes_accepts_t_dict_and_kwargs_mapping() -> None:
        """Test metadata attributes accepts t dict and kwargs mapping."""
        from_root = m.Metadata.model_validate({"attributes": m.Dict(root={"a": 1})})
        from_kwargs = m.Metadata(attributes={"b": 2})
        tm.that(from_root.attributes, eq={"a": 1})
        tm.that(from_kwargs.attributes, eq={"b": 2})

    @staticmethod
    @pytest.mark.parametrize("bad_value", [123, "text", 4.5, ["a", "b"]])
    def test_metadata_attributes_non_mapping_rejected(
        bad_value: t.Scalar | Sequence[str],
    ) -> None:
        """Test metadata attributes non mapping rejected."""
        tm.that(
            c.ValidationError,
            where=_is_same(m.ValidationError),
            msg=(
                f"constants binding {c.ValidationError!r} ({id(c.ValidationError)}) "
                f"differs from model SDK binding {m.ValidationError!r} "
                f"({id(m.ValidationError)})"
            ),
        )
        with pytest.raises(c.ValidationError) as failure:
            m.Metadata.model_validate({"attributes": bad_value})
        tm.that(failure.value.errors()[0]["loc"], eq=("attributes",))
        tm.that(failure.value.errors()[0]["type"], eq="dict_type")

    @staticmethod
    def test_metadata_attributes_broken_dump_object_rejected() -> None:
        """Test metadata attributes broken dump object rejected."""
        with pytest.raises(TypeError):
            m.Metadata.model_validate({"attributes": m.Tests.BrokenDumpModel()})

    # --- Metadata defaults and immutability ------------------------------

    @staticmethod
    def test_metadata_defaults_are_populated() -> None:
        """Test metadata defaults are populated."""
        model = m.Metadata()
        tm.that(model.created_at.tzinfo, eq=u.configured_timezone())
        tm.that(model.version, eq=c.METADATA_SCHEMA_VERSION_DEFAULT)
        tm.that(model.tags, truth=False)
        tm.that(isinstance(model.tags, tuple), truth=True)
        tm.that(model.created_by, none=True)
        tm.that(model.attributes, eq={})

    @staticmethod
    def test_metadata_is_frozen() -> None:
        """Test metadata is frozen."""
        model = m.Metadata()
        tm.rejects_assignment(model, "version", "2.0.0", expected=c.ValidationError)

    @staticmethod
    def test_metadata_rejects_extra_fields() -> None:
        """Test metadata rejects extra fields."""
        with pytest.raises(c.ValidationError):
            m.Metadata.model_validate({"unknown_field": 1})

    # --- FrozenValueModel value semantics --------------------------------

    def test_frozen_value_equal_by_value(self) -> None:
        """Test frozen value equal by value."""
        left = self._FrozenValue(name="item", count=1)
        right = self._FrozenValue(name="item", count=1)
        tm.that(left, eq=right)
        tm.that(hash(left), eq=hash(right))

    def test_frozen_value_differs_when_fields_differ(self) -> None:
        """Test frozen value differs when fields differ."""
        left = self._FrozenValue(name="item", count=1)
        other = self._FrozenValue(name="item", count=2)
        tm.that(left, ne=other)

    def test_frozen_value_not_equal_to_foreign_type(self) -> None:
        """Test frozen value not equal to foreign type."""
        tm.that(self._FrozenValue(name="item", count=1), ne="item")

    def test_frozen_value_is_hashable_in_set(self) -> None:
        """Test frozen value is hashable in set."""
        a = self._FrozenValue(name="x", count=1)
        b = self._FrozenValue(name="x", count=1)
        d = self._FrozenValue(name="y", count=2)
        tm.that(len({a, b, d}), eq=2)

    # --- IdentifiableMixin -----------------------------------------------

    def test_identifiable_autogenerates_unique_id(self) -> None:
        """Test identifiable autogenerates unique id."""
        first = self._Identifiable()
        second = self._Identifiable()
        tm.that(first.unique_id, ne=second.unique_id)
        tm.that(len(first.unique_id), eq=36)

    def test_identifiable_blank_unique_id_rejected(self) -> None:
        """Test identifiable blank unique id rejected."""
        with pytest.raises(c.ValidationError) as failure:
            self._Identifiable(unique_id="   ")
        tm.that(failure.value.errors()[0]["type"], eq="string_too_short")

    # --- TimestampableMixin ----------------------------------------------

    def test_timestampable_normalizes_to_configured_timezone(self) -> None:
        """Timestamp normalization follows the configured timezone owner."""
        moment = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        model = self._Timestampable(created_at=moment, updated_at=moment)
        tm.that(model.created_at.tzinfo, eq=u.configured_timezone())

    def test_timestampable_serializes_timestamps_as_iso_strings(self) -> None:
        """Test timestampable serializes timestamps as iso strings."""
        moment = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        model = self._Timestampable(created_at=moment)
        dumped = model.model_dump(mode="json")
        serialized = dumped["created_at"]
        tm.that(isinstance(serialized, str), truth=True)
        tm.that(datetime.fromisoformat(serialized), eq=moment)
        tm.that(dumped.get("updated_at"), none=True)

    def test_timestampable_rejects_updated_before_created(self) -> None:
        """Test timestampable rejects updated before created."""
        created = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
        earlier = created - timedelta(days=1)
        with pytest.raises(c.ValidationError):
            self._Timestampable(created_at=created, updated_at=earlier)

    # --- TimestampedModel -------------------------------------------------

    @staticmethod
    def test_timestamped_model_defaults_to_configured_timezone() -> None:
        """Default timestamps use the runtime's configured timezone."""
        model = m.TimestampedModel()
        tm.that(model.created_at.tzinfo, eq=u.configured_timezone())
        tm.that(model.updated_at, none=True)
