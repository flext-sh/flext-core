"""Behavioral contract tests for the internal CQRS model tier.

Exercises the public surface of ``m.Command``, ``m.Query`` and
``m.Pagination`` (message-type discriminators, identifier generation,
default state, pagination coercion and computed fields, immutability and
validation error paths). Every assertion targets observable behaviour a
caller depends on - never private attributes or implementation internals.
"""

from __future__ import annotations

from typing import Annotated

import pytest

from flext_core import c, p, t
from tests import m


class TestsFlextCoreCqrs:
    """Public-contract behaviour of the CQRS command/query/pagination models."""

    # ------------------------------------------------------------------ #
    # Command
    # ------------------------------------------------------------------ #
    def test_command_message_type_discriminator_is_command(self) -> None:
        # Act
        command = m.Command()

        # Assert
        assert command.message_type == "command"

    def test_command_message_type_is_immutable(self) -> None:
        # Arrange
        command = m.Command()

        # Act / Assert
        with pytest.raises(c.ValidationError):
            setattr(command, "message_type", "query")

    def test_command_generates_prefixed_identifier_by_default(self) -> None:
        # Act
        command = m.Command()

        # Assert
        assert command.command_id.startswith("cmd_")

    def test_command_identifiers_are_unique_per_instance(self) -> None:
        # Act
        first = m.Command()
        second = m.Command()

        # Assert
        assert first.command_id != second.command_id

    def test_command_accepts_explicit_identifier(self) -> None:
        # Act
        command = m.Command(command_id="cmd-supplied")

        # Assert
        assert command.command_id == "cmd-supplied"

    def test_command_issuer_id_defaults_to_none(self) -> None:
        # Act / Assert
        assert m.Command().issuer_id is None

    def test_command_retains_supplied_issuer_id(self) -> None:
        # Act
        command = m.Command(issuer_id="principal-42")

        # Assert
        assert command.issuer_id == "principal-42"

    def test_command_dump_exposes_public_fields_only(self) -> None:
        # Act
        dumped = m.Command(command_id="cmd-1").model_dump()

        # Assert
        assert dumped == {
            "message_type": "command",
            "command_type": dumped["command_type"],
            "command_id": "cmd-1",
            "issuer_id": None,
        }

    def test_command_subclass_extends_fields_and_inherits_contract(self) -> None:
        # Arrange
        class CreateItem(m.Command):
            name: Annotated[t.NonEmptyStr, m.Field(description="Item display name")]

        # Act
        command = CreateItem(name="widget")

        # Assert
        assert command.name == "widget"
        assert command.message_type == "command"
        assert command.command_id.startswith("cmd_")

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def test_query_message_type_discriminator_is_query(self) -> None:
        # Act / Assert
        assert m.Query().message_type == "query"

    def test_query_message_type_is_immutable(self) -> None:
        # Arrange
        query = m.Query()

        # Act / Assert
        with pytest.raises(c.ValidationError):
            setattr(query, "message_type", "command")

    def test_query_generates_prefixed_identifier_by_default(self) -> None:
        # Act / Assert
        assert m.Query().query_id.startswith("query_")

    def test_query_filters_default_to_empty(self) -> None:
        # Act / Assert
        assert dict(m.Query().filters) == {}

    def test_query_retains_supplied_filters(self) -> None:
        # Act
        query = m.Query(filters={"status": "active", "tenant": "acme"})

        # Assert
        assert dict(query.filters) == {"status": "active", "tenant": "acme"}

    def test_query_query_type_defaults_to_none(self) -> None:
        # Act / Assert
        assert m.Query().query_type is None

    def test_query_provides_default_pagination(self) -> None:
        # Act
        query = m.Query()

        # Assert
        assert query.pagination.page == 1

    def test_query_coerces_pagination_mapping_into_model(self) -> None:
        # Act
        query = m.Query(pagination={"page": 2, "size": 5})

        # Assert
        assert query.pagination.page == 2
        assert query.pagination.size == 5
        assert query.pagination.offset == 5

    @pytest.mark.parametrize(
        "invalid_pagination",
        [
            {"page": 0, "size": -5},
            {"size": 10_000},
            {"page": -1},
        ],
    )
    def test_query_falls_back_to_default_pagination_on_invalid_input(
        self,
        invalid_pagination: t.MappingKV[str, t.Scalar],
    ) -> None:
        # Act
        query = m.Query(pagination=invalid_pagination)

        # Assert
        assert query.pagination.page == 1
        assert query.pagination.size == 10

    def test_query_subclass_extends_fields_and_inherits_contract(self) -> None:
        # Arrange
        class GetItem(m.Query):
            item_id: Annotated[t.NonEmptyStr, m.Field(description="Target item id")]

        # Act
        query = GetItem(item_id="7")

        # Assert
        assert query.item_id == "7"
        assert query.message_type == "query"
        assert query.query_id.startswith("query_")

    # ------------------------------------------------------------------ #
    # Pagination
    # ------------------------------------------------------------------ #
    def test_pagination_defaults_to_first_page(self) -> None:
        # Act
        pagination = m.Pagination()

        # Assert
        assert pagination.page == 1
        assert pagination.offset == 0

    def test_pagination_limit_mirrors_size(self) -> None:
        # Act
        pagination = m.Pagination(size=25)

        # Assert
        assert pagination.limit == 25

    @pytest.mark.parametrize(
        ("page", "size", "expected_offset"),
        [
            (1, 10, 0),
            (2, 10, 10),
            (3, 20, 40),
            (5, 50, 200),
        ],
    )
    def test_pagination_offset_is_derived_from_page_and_size(
        self,
        page: int,
        size: int,
        expected_offset: int,
    ) -> None:
        # Act
        pagination = m.Pagination(page=page, size=size)

        # Assert
        assert pagination.offset == expected_offset

    @pytest.mark.parametrize(
        "invalid_kwargs",
        [
            {"size": c.MAX_PAGE_SIZE + 1},
            {"page": 0},
            {"page": -3},
            {"size": 0},
        ],
    )
    def test_pagination_rejects_out_of_range_values(
        self,
        invalid_kwargs: t.MappingKV[str, int],
    ) -> None:
        # Act / Assert
        with pytest.raises(c.ValidationError):
            m.Pagination(**invalid_kwargs)
