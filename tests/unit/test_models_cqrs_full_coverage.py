"""Behavioral tests for the FlextModels CQRS public contract.

These tests exercise only the observable public surface of the CQRS models
(``Pagination``, ``Query``, ``Command``, ``Event``, ``Handler`` and the
``FlextMessage`` discriminated union) through the ``m`` / ``c`` facades. No
private attributes, internal collaborators, or module internals are touched.

Copyright (c) 2026 FLEXT Team. All rights reserved.
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

from typing import cast

import pytest

from tests import c, m, t, u


class TestsFlextCoreModelsCqrs:
    """Public-contract behavior of the CQRS models."""

    # ------------------------------------------------------------------ #
    # Pagination                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_pagination_applies_documented_defaults() -> None:
        """Test pagination applies documented defaults."""
        page = m.Pagination()

        assert page.page == c.FIRST_PAGE_INDEX
        assert page.size == c.DEFAULT_PAGE_SIZE

    @staticmethod
    @pytest.mark.parametrize(
        ("page", "size"),
        [
            (c.FIRST_PAGE_INDEX, c.MIN_PAGE_SIZE),
            (c.FIRST_PAGE_INDEX + 2, c.DEFAULT_PAGE_SIZE),
            (c.FIRST_PAGE_INDEX + 1, c.MAX_PAGE_SIZE),
            (c.FIRST_PAGE_INDEX + 9, c.DEFAULT_PAGE_SIZE),
        ],
    )
    def test_pagination_limit_equals_size(page: int, size: int) -> None:
        """Test pagination limit equals size."""
        assert m.Pagination(page=page, size=size).limit == size

    @staticmethod
    @pytest.mark.parametrize(
        ("completed_pages", "size"),
        [
            (0, c.MIN_PAGE_SIZE),
            (1, c.DEFAULT_PAGE_SIZE),
            (2, c.MAX_PAGE_SIZE),
            (4, c.DEFAULT_PAGE_SIZE),
        ],
    )
    def test_pagination_offset_is_derived_from_page_and_size(
        completed_pages: int,
        size: int,
    ) -> None:
        """Test pagination offset is derived from page and size."""
        pagination = m.Pagination(page=c.FIRST_PAGE_INDEX + completed_pages, size=size)
        assert pagination.offset == completed_pages * size

    @staticmethod
    def test_pagination_exposes_computed_fields_in_dump() -> None:
        """Test pagination exposes computed fields in dump."""
        dumped = m.Pagination(
            page=c.FIRST_PAGE_INDEX + 1,
            size=c.DEFAULT_PAGE_SIZE,
        ).model_dump()

        assert dumped["limit"] == c.DEFAULT_PAGE_SIZE
        assert dumped["offset"] == c.DEFAULT_PAGE_SIZE

    @staticmethod
    @pytest.mark.parametrize(
        "size",
        [c.MIN_PAGE_SIZE - 1, c.MAX_PAGE_SIZE + 1, c.MAX_PAGE_SIZE + 2],
    )
    def test_pagination_rejects_size_outside_configured_bounds(size: int) -> None:
        """Configured pagination bounds reject values on either side."""
        with pytest.raises(m.ValidationError):
            m.Pagination(size=size)

    @staticmethod
    @pytest.mark.parametrize(
        ("page", "size"),
        [(0, c.MIN_PAGE_SIZE), (c.FIRST_PAGE_INDEX, 0), (-1, c.MIN_PAGE_SIZE)],
    )
    def test_pagination_rejects_non_positive_bounds(page: int, size: int) -> None:
        """Test pagination rejects non positive bounds."""
        with pytest.raises(m.ValidationError):
            m.Pagination(page=page, size=size)

    # ------------------------------------------------------------------ #
    # Query pagination coercion                                          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_query_defaults_to_a_pagination_instance() -> None:
        """Test query defaults to a pagination instance."""
        query = m.Query()

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == c.FIRST_PAGE_INDEX
        assert query.pagination.size == c.DEFAULT_PAGE_SIZE
        assert query.message_type == "query"

    @staticmethod
    def test_query_coerces_pagination_mapping_including_stringified_ints() -> None:
        """Test query coerces pagination mapping including stringified ints."""
        query = m.Query.model_validate({
            "pagination": {
                "page": str(c.FIRST_PAGE_INDEX + 3),
                "size": str(c.DEFAULT_PAGE_SIZE),
            },
            "filters": {},
        })

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == c.FIRST_PAGE_INDEX + 3
        assert query.pagination.size == c.DEFAULT_PAGE_SIZE

    @staticmethod
    def test_query_uses_default_pagination_when_none_supplied() -> None:
        """Test query uses default pagination when none supplied."""
        query = m.Query.model_validate({"pagination": None, "filters": {}})

        assert isinstance(query.pagination, m.Pagination)
        assert query.pagination.page == c.FIRST_PAGE_INDEX

    @staticmethod
    def test_query_rejects_invalid_pagination() -> None:
        """Invalid explicit pagination retains its native validation failure."""
        with pytest.raises(m.ValidationError):
            m.Query.model_validate({
                "pagination": {"size": c.MAX_PAGE_SIZE + 1},
                "filters": {},
            })

    @staticmethod
    def test_query_subclass_pagination_override_is_honored() -> None:
        """Test query subclass pagination override is honored."""

        class CustomQuery(m.Query):
            class Pagination(m.Pagination):
                pass

        query = CustomQuery.model_validate({
            "pagination": {"page": c.FIRST_PAGE_INDEX, "size": c.MIN_PAGE_SIZE},
            "filters": {},
        })

        assert type(query.pagination) is CustomQuery.Pagination
        assert query.pagination.page == c.FIRST_PAGE_INDEX
        assert query.pagination.size == c.MIN_PAGE_SIZE

    # ------------------------------------------------------------------ #
    # Command / Event identity                                           #
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_command_has_default_type_and_generated_id() -> None:
        """Test command has default type and generated id."""
        command = m.Command()

        assert command.message_type == "command"
        assert command.command_type == c.DEFAULT_COMMAND_TYPE
        assert command.command_id.startswith("cmd_")

    @staticmethod
    def test_command_generates_unique_ids() -> None:
        """Test command generates unique ids."""
        assert m.Command().command_id != m.Command().command_id

    @staticmethod
    def test_command_accepts_explicit_type() -> None:
        """Test command accepts explicit type."""
        assert m.Command(command_type="run").command_type == "run"

    @staticmethod
    def test_event_has_event_type_and_generated_id() -> None:
        """Test event has event type and generated id."""
        event = m.Event(event_type="created", aggregate_id="agg-1")

        assert event.message_type == "event"
        assert event.event_type == "created"
        assert event.event_id.startswith("evt_")

    # ------------------------------------------------------------------ #
    # Handler                                                            #
    # ------------------------------------------------------------------ #
    @staticmethod
    def test_handler_stores_identity_and_defaults_mode_to_command() -> None:
        """Test handler stores identity and defaults mode to command."""
        handler = m.Handler(
            handler_type=c.HandlerType.QUERY,
            handler_id="h-1",
            handler_name="handler",
        )

        assert handler.handler_id == "h-1"
        assert handler.handler_name == "handler"
        assert handler.handler_type == c.HandlerType.QUERY
        assert handler.handler_mode == c.HandlerType.COMMAND

    @staticmethod
    @pytest.mark.parametrize(
        "handler_type",
        [c.HandlerType.COMMAND, c.HandlerType.QUERY, c.HandlerType.EVENT],
    )
    def test_handler_accepts_each_handler_type(
        handler_type: c.HandlerType,
    ) -> None:
        """Test handler accepts each handler type."""
        handler = m.Handler(handler_type=handler_type, handler_id="h", handler_name="n")

        assert handler.handler_type == handler_type

    @staticmethod
    def test_handler_rejects_empty_identity() -> None:
        """Test handler rejects empty identity."""
        with pytest.raises(m.ValidationError):
            m.Handler(
                handler_type=c.HandlerType.COMMAND,
                handler_id="",
                handler_name="",
            )

    # ------------------------------------------------------------------ #
    # FlextMessage discriminated union                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    @pytest.mark.parametrize(
        ("payload", "expected_cls"),
        [
            ({"message_type": "command", "command_type": "run"}, m.Command),
            ({"message_type": "query", "filters": {}}, m.Query),
            (
                {
                    "message_type": "event",
                    "event_type": "created",
                    "aggregate_id": "agg-1",
                },
                m.Event,
            ),
        ],
    )
    def test_flext_message_union_discriminates_on_message_type(
        payload: t.MappingKV[str, str | t.MappingKV[str, str]],
        expected_cls: type,
    ) -> None:
        """Test flext message union discriminates on message type."""
        adapter = u.type_adapter(m.FlextMessage.__value__)

        parsed = adapter.validate_python(payload)

        assert isinstance(parsed, expected_cls)
        message = cast("m.Command | m.Query | m.Event", parsed)
        assert message.message_type == payload["message_type"]
