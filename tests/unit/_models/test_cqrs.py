"""Behavioral contract tests for the internal CQRS model tier.

Exercises the public surface of ``m.Command``, ``m.Query`` and
``m.Pagination`` (message-type discriminators, identifier generation,
default state, pagination coercion and computed fields, immutability and
validation error paths). Every assertion targets observable behaviour a
caller depends on - never private attributes or implementation internals.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import Annotated

import pytest

from flext_core import c, t
from tests import m, tm


class TestsFlextCoreCqrs:
    """Public-contract behaviour of the CQRS command/query/pagination models."""

    # ------------------------------------------------------------------ #
    # Command
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_command_message_type_discriminator_is_command() -> None:
        # Act
        command = m.Command()

        # Assert
        tm.that(command.message_type, eq="command")

    @staticmethod
    def test_command_message_type_is_immutable() -> None:
        # Arrange
        command = m.Command()

        # Act / Assert
        tm.rejects_assignment(
            command,
            "message_type",
            "query",
            expected=c.ValidationError,
        )

    @staticmethod
    def test_command_generates_prefixed_identifier_by_default() -> None:
        # Act
        command = m.Command()

        # Assert
        tm.that(command.command_id.startswith("cmd_"), truth=True)

    @staticmethod
    def test_command_identifiers_are_unique_per_instance() -> None:
        # Act
        first = m.Command()
        second = m.Command()

        # Assert
        tm.that(first.command_id, ne=second.command_id)

    @staticmethod
    def test_command_accepts_explicit_identifier() -> None:
        # Act
        command = m.Command(command_id="cmd-supplied")

        # Assert
        tm.that(command.command_id, eq="cmd-supplied")

    @staticmethod
    def test_command_issuer_id_defaults_to_none() -> None:
        # Act / Assert
        tm.that(m.Command().issuer_id, none=True)

    @staticmethod
    def test_command_retains_supplied_issuer_id() -> None:
        # Act
        command = m.Command(issuer_id="principal-42")

        # Assert
        tm.that(command.issuer_id, eq="principal-42")

    @staticmethod
    def test_command_dump_exposes_public_fields_only() -> None:
        # Act
        dumped = m.Command(command_id="cmd-1").model_dump()

        # Assert
        tm.that(
            dumped,
            eq={
                "message_type": "command",
                "command_type": dumped["command_type"],
                "command_id": "cmd-1",
                "issuer_id": None,
            },
        )

    @staticmethod
    def test_command_subclass_extends_fields_and_inherits_contract() -> None:
        # Arrange
        class CreateItem(m.Command):
            name: Annotated[t.NonEmptyStr, m.Field(description="Item display name")]

        # Act
        command = CreateItem(name="widget")

        # Assert
        tm.that(command.name, eq="widget")
        tm.that(command.message_type, eq="command")
        tm.that(command.command_id.startswith("cmd_"), truth=True)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_query_message_type_discriminator_is_query() -> None:
        # Act / Assert
        tm.that(m.Query().message_type, eq="query")

    @staticmethod
    def test_query_message_type_is_immutable() -> None:
        # Arrange
        query = m.Query()

        # Act / Assert
        tm.rejects_assignment(
            query,
            "message_type",
            "command",
            expected=c.ValidationError,
        )

    @staticmethod
    def test_query_generates_prefixed_identifier_by_default() -> None:
        # Act / Assert
        tm.that(m.Query().query_id.startswith("query_"), truth=True)

    @staticmethod
    def test_query_filters_default_to_empty() -> None:
        # Act / Assert
        tm.that(dict(m.Query().filters), eq={})

    @staticmethod
    def test_query_retains_supplied_filters() -> None:
        # Act
        query = m.Query(filters={"status": "active", "tenant": "acme"})

        # Assert
        tm.that(dict(query.filters), eq={"status": "active", "tenant": "acme"})

    @staticmethod
    def test_query_query_type_defaults_to_none() -> None:
        # Act / Assert
        tm.that(m.Query().query_type, none=True)

    @staticmethod
    def test_query_provides_default_pagination() -> None:
        # Act
        query = m.Query()

        # Assert
        tm.that(query.pagination.page, eq=1)

    @staticmethod
    def test_query_coerces_pagination_mapping_into_model() -> None:
        # Act
        query = m.Query.model_validate({"pagination": {"page": 2, "size": 5}})

        # Assert
        tm.that(query.pagination.page, eq=2)
        tm.that(query.pagination.size, eq=5)
        tm.that(query.pagination.offset, eq=5)

    @pytest.mark.parametrize(
        "invalid_pagination",
        [{"page": 0, "size": -5}, {"size": c.MAX_PAGE_SIZE + 1}, {"page": -1}],
    )
    @staticmethod
    def test_query_rejects_invalid_pagination(
        invalid_pagination: t.MappingKV[str, t.Scalar],
    ) -> None:
        with pytest.raises(c.ValidationError):
            m.Query.model_validate({"pagination": invalid_pagination})

    @staticmethod
    def test_query_subclass_extends_fields_and_inherits_contract() -> None:
        # Arrange
        class GetItem(m.Query):
            item_id: Annotated[t.NonEmptyStr, m.Field(description="Target item id")]

        # Act
        query = GetItem(item_id="7")

        # Assert
        tm.that(query.item_id, eq="7")
        tm.that(query.message_type, eq="query")
        tm.that(query.query_id.startswith("query_"), truth=True)

    # ------------------------------------------------------------------ #
    # Pagination
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_pagination_defaults_to_first_page() -> None:
        # Act
        pagination = m.Pagination()

        # Assert
        tm.that(pagination.page, eq=1)
        tm.that(pagination.offset, eq=0)

    @staticmethod
    def test_pagination_limit_mirrors_size() -> None:
        # Act
        pagination = m.Pagination(size=25)

        # Assert
        tm.that(pagination.limit, eq=25)

    @pytest.mark.parametrize(
        ("page", "size", "expected_offset"),
        [(1, 10, 0), (2, 10, 10), (3, 20, 40), (5, 50, 200)],
    )
    @staticmethod
    def test_pagination_offset_is_derived_from_page_and_size(
        page: int,
        size: int,
        expected_offset: int,
    ) -> None:
        # Act
        pagination = m.Pagination(page=page, size=size)

        # Assert
        tm.that(pagination.offset, eq=expected_offset)

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [{"size": c.MAX_PAGE_SIZE + 1}, {"page": 0}, {"page": -3}, {"size": 0}],
    )
    @staticmethod
    def test_pagination_rejects_out_of_range_values(
        invalid_kwargs: t.MappingKV[str, int],
    ) -> None:
        # Act / Assert
        with pytest.raises(c.ValidationError):
            m.Pagination(**invalid_kwargs)
