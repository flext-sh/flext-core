"""Behavioral tests for the tests typing facade (``t``) public surface.

Exercises the observable contract callers depend on:
- ``flext_core`` keeps shared ``TypeVar`` / ``ParamSpec`` helpers out of its
  public surface,
- the runtime type-check tuples (``PRIMITIVES_TYPES`` etc.) expose the exact
  membership documented for them and actually classify values via
  ``isinstance``.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, ParamSpec, TypeVar

import pydantic

import flext_core
from tests import c, m, p, t, tm, u


class TestsFlextCoreTypingsAliases:
    """Public contract of the ``t`` typing facade and its type-check tuples."""

    @staticmethod
    def test_flext_core_hides_shared_generic_helpers() -> None:
        """flext_core must not expose shared TypeVar/ParamSpec helpers publicly."""
        helpers = [
            name
            for name in flext_core.__all__
            if isinstance(getattr(flext_core, name), (TypeVar, ParamSpec))
        ]
        tm.that(helpers, eq=[])

    @staticmethod
    def test_primitives_types_membership() -> None:
        """PRIMITIVES_TYPES is exactly (str, int, float, bool)."""
        tm.that(c.PRIMITIVES_TYPES, eq=(str, int, float, bool))

    @staticmethod
    def test_numeric_types_membership() -> None:
        """NUMERIC_TYPES is exactly (int, float)."""
        tm.that(c.NUMERIC_TYPES, eq=(int, float))

    @staticmethod
    def test_scalar_types_membership() -> None:
        """SCALAR_TYPES is exactly (str, int, float, bool, datetime)."""
        tm.that(c.SCALAR_TYPES, eq=(str, int, float, bool, datetime))

    @staticmethod
    def test_container_types_membership() -> None:
        """CONTAINER_TYPES extends the scalar set with Path."""
        tm.that(c.CONTAINER_TYPES, eq=(str, int, float, bool, datetime, Path))

    @staticmethod
    def test_container_and_collection_types_include_collections() -> None:
        """CONTAINER_AND_COLLECTION_TYPES adds list/dict/tuple to CONTAINER_TYPES."""
        tm.that(
            c.CONTAINER_AND_COLLECTION_TYPES,
            eq=(str, int, float, bool, datetime, Path, list, dict, tuple),
        )

    @staticmethod
    @c.Tests.MARK.parametrize(
        ("value", "is_primitive", "is_numeric", "is_scalar"),
        [
            ("text", True, False, True),
            (7, True, True, True),
            (1.5, True, True, True),
            (True, True, True, True),
            (datetime(2026, 1, 1, tzinfo=UTC), False, False, True),
            (Path(tempfile.gettempdir()), False, False, False),
            ([1, 2], False, False, False),
        ],
    )
    def test_type_check_tuples_classify_values(
        value: p.AttributeProbe,
        *,
        is_primitive: bool,
        is_numeric: bool,
        is_scalar: bool,
    ) -> None:
        """The type-check tuples classify values correctly via isinstance."""
        tm.that(isinstance(value, c.PRIMITIVES_TYPES), eq=is_primitive)
        tm.that(isinstance(value, c.NUMERIC_TYPES), eq=is_numeric)
        tm.that(isinstance(value, c.SCALAR_TYPES), eq=is_scalar)

    @staticmethod
    @c.Tests.MARK.parametrize(
        "value",
        [
            "text",
            3,
            2.0,
            True,
            datetime(2026, 1, 1, tzinfo=UTC),
            Path(tempfile.gettempdir()),
            [1],
            {"a": 1},
            (1,),
        ],
    )
    def test_container_and_collection_tuple_accepts_every_container_value(
        value: p.AttributeProbe,
    ) -> None:
        """Every documented container/collection value is recognised by the tuple."""
        tm.ok(isinstance(value, c.CONTAINER_AND_COLLECTION_TYPES))

    @staticmethod
    def test_port_resolves_pydantic_json_value_alias() -> None:
        """A port over ``pydantic.JsonValue`` validates through its alias.

        ``JsonValue`` is a ``typing_extensions.TypeAliasType``, a different
        class from ``typing.TypeAliasType`` on Python 3.13; declaring the port
        must resolve the alias instead of handing it to ``isinstance``.
        """

        class Holder(m.BaseModel):
            """Model declaring the port over pydantic's JSON alias."""

            value: t.Port[pydantic.JsonValue]

        tm.that(Holder(value="texto").value, eq="texto")
        with u.Tests.raises(m.ValidationError):
            Holder.model_validate({"value": object()})

    @staticmethod
    @c.Tests.MARK.parametrize("mode", ["validation", "serialization"])
    def test_shared_skipped_alias_is_omitted_from_json_schema(
        mode: Literal["validation", "serialization"],
    ) -> None:
        """Two fields sharing one skipped alias leave the JSON Schema, not break it.

        The alias becomes one shared schema definition (the FlextCli prompt
        readers declare it twice); the schema keeps every other field.
        """

        class Reader:
            """Opaque collaborator with no JSON representation."""

        class Holder(m.BaseModel):
            """Model declaring the same skipped alias on two fields."""

            model_config = m.ConfigDict(arbitrary_types_allowed=True)

            first: t.SkipJsonSchema[Reader] = m.Field(
                default_factory=Reader, exclude=True
            )
            second: t.SkipJsonSchema[Reader] = m.Field(
                default_factory=Reader, exclude=True
            )
            label: str = "visible"

        schema = Holder.model_json_schema(mode=mode)
        tm.that(sorted(schema["properties"]), eq=["label"])
