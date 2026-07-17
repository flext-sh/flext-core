"""Behavioral tests for the FlextModels CQRS public contract.

These tests exercise only the observable public surface of the CQRS models
(``Pagination``, ``Query``, ``Command``, ``Event``, ``Handler`` and the
``FlextMessage`` discriminated union) through the ``m`` / ``c`` facades. No
private attributes, internal collaborators, or module internals are touched.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tests import c
from tests import m


class TestsFlextCoreModelsCqrs:
    """Public-contract behavior of the CQRS models."""

    # ------------------------------------------------------------------ #
    # Pagination                                                         #
    # ------------------------------------------------------------------ #
    def test_pagination_applies_documented_defaults(self) -> None:
        page = m.Pagination()

        assert page.page == c.DEFAULT_RETRY_DELAY_SECONDS
        assert page.size == c.DEFAULT_PAGE_SIZE

    @pytest.mark.parametrize(
        ("page", "size"),
        [(1, 10), (3, 11), (2, 50), (10, 100)],
    )
    def test_pagination_limit_equals_size(self, page: int, size: int) -> None:
        assert m.Pagination(page=page, size=size).limit == size

    @pytest.mark.parametrize(
        ("page", "size", "expected_offset"),
        [(1, 10, 0), (2, 10, 10), (3, 11, 22), (5, 20, 80)],
    )
    def test_pagination_offset_is_derived_from_page_and_size(
        self,
        page: int,
        size: int,
        expected_offset: int,
    ) -> None:
        assert m.Pagination(page=page, size=size).offset == expected_offset

    def test_pagination_exposes_computed_fields_in_dump(self) -> None:
        dumped = m.Pagination(page=2, size=20).model_dump()

        assert dumped["limit"] == 20
        assert dumped["offset"] == 20

    @pytest.mark.parametrize("size", [c.MAX_PAGE_SIZE + 1, 5000])
    def test_pagination_rejects_size_above_maximum(self, size: int) -> None:
        with pytest.raises(ValidationError):
            m.Pagination(size=size)

    @pytest.mark.parametrize(("page", "size"), [(0, 10), (1, 0), (-1, 10)])
    def test_pagination_rejects_non_positive_bounds(
        self,
        page: int,
        size: int,
    ) -> None:
        with pytest.raises(ValidationError):
            m.Pagination(page=page, size=size)

    # ------------------------------------------------------------------ #
    # Query pagination coercion                                          #
    # ------------------------------------------------------------------ #
    def test_query_defaults_to_a_pagination_instance(self) -> None:
        query = m.Query()

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == c.DEFAULT_RETRY_DELAY_SECONDS
        assert query.pagination.size == c.DEFAULT_PAGE_SIZE
        assert query.message_type == "query"

    def test_query_coerces_pagination_mapping_including_stringified_ints(
        self,
    ) -> None:
        query = m.Query.model_validate({
            "pagination": {"page": "4", "size": "20"},
            "filters": {},
        })

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == 4
        assert query.pagination.size == 20

    def test_query_uses_default_pagination_when_none_supplied(self) -> None:
        query = m.Query.model_validate({"pagination": None, "filters": {}})

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == c.DEFAULT_RETRY_DELAY_SECONDS

    def test_query_falls_back_to_default_pagination_on_invalid_input(self) -> None:
        query = m.Query.model_validate({
            "pagination": {"size": c.MAX_PAGE_SIZE + 1},
            "filters": {},
        })

        # Invalid pagination is a recoverable condition: the query keeps a
        # valid default Pagination rather than propagating the error.
        assert query.pagination.size == c.DEFAULT_PAGE_SIZE

    def test_query_subclass_pagination_override_is_honored(self) -> None:
        class CustomQuery(m.Query):
            class Pagination(m.Pagination):
                pass

        query = CustomQuery.model_validate({
            "pagination": {"page": 1, "size": 9},
            "filters": {},
        })

        assert type(query.pagination) is CustomQuery.Pagination
        assert query.pagination.page == 1
        assert query.pagination.size == 9

    # ------------------------------------------------------------------ #
    # Command / Event identity                                           #
    # ------------------------------------------------------------------ #
    def test_command_has_default_type_and_generated_id(self) -> None:
        command = m.Command()

        assert command.message_type == "command"
        assert command.command_type == c.DEFAULT_COMMAND_TYPE
        assert command.command_id.startswith("cmd_")

    def test_command_generates_unique_ids(self) -> None:
        assert m.Command().command_id != m.Command().command_id

    def test_command_accepts_explicit_type(self) -> None:
        assert m.Command(command_type="run").command_type == "run"

    def test_event_has_event_type_and_generated_id(self) -> None:
        event = m.Event(event_type="created", aggregate_id="agg-1")

        assert event.message_type == "event"
        assert event.event_type == "created"
        assert event.event_id.startswith("evt_")

    # ------------------------------------------------------------------ #
    # Handler                                                            #
    # ------------------------------------------------------------------ #
    def test_handler_stores_identity_and_defaults_mode_to_command(self) -> None:
        handler = m.Handler(
            handler_type=c.HandlerType.QUERY,
            handler_id="h-1",
            handler_name="handler",
        )

        assert handler.handler_id == "h-1"
        assert handler.handler_name == "handler"
        assert handler.handler_type == c.HandlerType.QUERY
        assert handler.handler_mode == c.HandlerType.COMMAND

    @pytest.mark.parametrize(
        "handler_type",
        [c.HandlerType.COMMAND, c.HandlerType.QUERY, c.HandlerType.EVENT],
    )
    def test_handler_accepts_each_handler_type(
        self,
        handler_type: c.HandlerType,
    ) -> None:
        handler = m.Handler(
            handler_type=handler_type,
            handler_id="h",
            handler_name="n",
        )

        assert handler.handler_type == handler_type

    def test_handler_rejects_empty_identity(self) -> None:
        with pytest.raises(ValidationError):
            m.Handler(
                handler_type=c.HandlerType.COMMAND,
                handler_id="",
                handler_name="",
            )

    # ------------------------------------------------------------------ #
    # FlextMessage discriminated union                                   #
    # ------------------------------------------------------------------ #
    @pytest.mark.parametrize(
        ("payload", "expected_cls"),
        [
            ({"message_type": "command", "command_type": "run"}, "Command"),
            ({"message_type": "query", "filters": {}}, "Query"),
            (
                {
                    "message_type": "event",
                    "event_type": "created",
                    "aggregate_id": "agg-1",
                },
                "Event",
            ),
        ],
    )
    def test_flext_message_union_discriminates_on_message_type(
        self,
        payload: dict[str, str | dict[str, str]],
        expected_cls: str,
    ) -> None:
        adapter = m.TypeAdapter(m.FlextMessage.__value__)

        parsed = adapter.validate_python(payload)

        assert type(parsed).__name__ == expected_cls
        assert parsed.message_type == payload["message_type"]
