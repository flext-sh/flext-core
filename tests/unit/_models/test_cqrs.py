"""Tests for CQRS models via FlextModels facade.

Covers Command, Query, Event, Pagination, Bus, Handler (with Builder),
and FlextMessage discriminated union.

Source: flext_core._models.cqrs (477 LOC)
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import cast

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from flext_tests import tm
from tests import c, m, t, u


class TestFlextModelsCqrs:
    """Tests for flext_core._models.cqrs via the m facade."""

    # ── Command ────────────────────────────────────────────────

    def test_command_defaults(self) -> None:
        cmd = m.Command()
        tm.that(cmd.message_type, eq="command")
        tm.that(cmd.command_type, eq=c.DEFAULT_COMMAND_TYPE)
        tm.that(cmd.command_id, starts="cmd_")
        tm.that(cmd.issuer_id, none=True)
        tm.that(cmd.tag, eq="command")

    def test_command_custom_fields(self) -> None:
        cmd = m.Command(
            command_type="create_user",
            command_id="cmd_test_123",
            issuer_id="user-42",
        )
        tm.that(cmd.command_type, eq="create_user")
        tm.that(cmd.command_id, eq="cmd_test_123")
        tm.that(cmd.issuer_id, eq="user-42")

    def test_command_event_type_always_none(self) -> None:
        cmd = m.Command()
        tm.that(cmd.event_type, none=True)

    def test_command_query_type_always_none(self) -> None:
        cmd = m.Command()
        tm.that(cmd.query_type, none=True)

    def test_command_unique_ids(self) -> None:
        cmd1 = m.Command()
        cmd2 = m.Command()
        tm.that(cmd1.command_id, ne=cmd2.command_id)

    def test_command_serialization_round_trip(self) -> None:
        cmd = m.Command(command_type="deploy", issuer_id="ci-bot")
        data = cmd.model_dump()
        tm.that(data["message_type"], eq="command")
        tm.that(data["command_type"], eq="deploy")
        tm.that(data["issuer_id"], eq="ci-bot")
        restored = m.Command.model_validate(data)
        tm.that(restored.command_type, eq=cmd.command_type)
        tm.that(restored.command_id, eq=cmd.command_id)

    def test_command_json_round_trip(self) -> None:
        cmd = m.Command(command_type="sync")
        json_str = cmd.model_dump_json()
        restored = m.Command.model_validate_json(json_str)
        tm.that(restored.command_type, eq="sync")
        tm.that(restored.command_id, eq=cmd.command_id)

    def test_command_message_type_frozen(self) -> None:
        with pytest.raises(ValidationError):
            m.Command.model_validate({"message_type": "query"})

    def test_command_rejects_empty_command_type(self) -> None:
        with pytest.raises(ValidationError):
            m.Command(command_type="")

    # ── Pagination ─────────────────────────────────────────────

    def test_pagination_defaults(self) -> None:
        page = m.Pagination()
        tm.that(page.page, eq=c.DEFAULT_RETRY_DELAY_SECONDS)
        tm.that(page.size, eq=c.DEFAULT_PAGE_SIZE)

    def test_pagination_computed_limit(self) -> None:
        page = m.Pagination(page=3, size=11)
        tm.that(page.limit, eq=11)

    def test_pagination_computed_offset(self) -> None:
        page = m.Pagination(page=3, size=20)
        tm.that(page.offset, eq=40)

    def test_pagination_page_one_offset_zero(self) -> None:
        page = m.Pagination(page=1, size=50)
        tm.that(page.offset, eq=0)

    @pytest.mark.parametrize(
        ("page", "size", "expected_offset"),
        [
            (1, 10, 0),
            (2, 10, 10),
            (5, 25, 100),
            (1, 100, 0),
            (10, 50, 450),
        ],
        ids=["page1", "page2", "page5-size25", "page1-size100", "page10-size50"],
    )
    def test_pagination_offset_parametrized(
        self, page: int, size: int, expected_offset: int
    ) -> None:
        p = m.Pagination(page=page, size=size)
        tm.that(p.offset, eq=expected_offset)

    def test_pagination_rejects_zero_page(self) -> None:
        with pytest.raises(ValidationError):
            m.Pagination(page=0)

    def test_pagination_rejects_zero_size(self) -> None:
        with pytest.raises(ValidationError):
            m.Pagination(size=0)

    def test_pagination_rejects_oversized(self) -> None:
        with pytest.raises(ValidationError):
            m.Pagination(size=c.MAX_PAGE_SIZE + 1)

    def test_pagination_serialization_round_trip(self) -> None:
        page = m.Pagination(page=2, size=25)
        data = page.model_dump()
        tm.that(data["page"], eq=2)
        tm.that(data["size"], eq=25)
        tm.that(data["limit"], eq=25)
        tm.that(data["offset"], eq=25)
        restored = m.Pagination.model_validate(data)
        tm.that(restored.page, eq=page.page)
        tm.that(restored.size, eq=page.size)

    # ── Query ──────────────────────────────────────────────────

    def test_query_defaults(self) -> None:
        query = m.Query()
        tm.that(query.message_type, eq="query")
        tm.that(query.query_id, starts="query_")
        tm.that(query.query_type, none=True)
        tm.that(query.tag, eq="query")

    def test_query_custom_fields(self) -> None:
        query = m.Query(
            query_type="get_users",
            filters=t.Dict(root={"status": "active"}),
        )
        tm.that(query.query_type, eq="get_users")
        tm.that(query.filters.root["status"], eq="active")

    def test_query_command_type_always_none(self) -> None:
        query = m.Query()
        tm.that(query.command_type, none=True)

    def test_query_event_type_always_none(self) -> None:
        query = m.Query()
        tm.that(query.event_type, none=True)

    def test_query_unique_ids(self) -> None:
        q1 = m.Query()
        q2 = m.Query()
        tm.that(q1.query_id, ne=q2.query_id)

    def test_query_message_type_frozen(self) -> None:
        with pytest.raises(ValidationError):
            m.Query.model_validate({"message_type": "command"})

    def test_query_serialization_round_trip(self) -> None:
        query = m.Query(query_type="list_items")
        data = query.model_dump()
        tm.that(data["message_type"], eq="query")
        tm.that(data["query_type"], eq="list_items")
        restored = m.Query.model_validate(data)
        tm.that(restored.query_type, eq=query.query_type)
        tm.that(restored.query_id, eq=query.query_id)

    def test_query_validate_pagination_from_dict(self) -> None:
        parsed = m.Query.model_validate({
            "pagination": {"page": 4, "size": 20},
            "filters": {},
        })
        tm.that(parsed.pagination, is_=m.Pagination)
        pag = cast("m.Pagination", parsed.pagination)
        tm.that(pag.page, eq=4)
        tm.that(pag.size, eq=20)

    def test_query_validate_pagination_from_string_coercion(self) -> None:
        parsed = m.Query.model_validate({
            "pagination": {"page": "3", "size": "15"},
            "filters": {},
        })
        tm.that(parsed.pagination, is_=m.Pagination)
        pag = cast("m.Pagination", parsed.pagination)
        tm.that(pag.page, eq=3)
        tm.that(pag.size, eq=15)

    def test_query_validate_pagination_none_defaults(self) -> None:
        defaulted = m.Query.model_validate({"pagination": None, "filters": {}})
        tm.that(defaulted.pagination, is_=m.Pagination)
        pag = cast("m.Pagination", defaulted.pagination)
        tm.that(pag.page, eq=c.DEFAULT_RETRY_DELAY_SECONDS)
        tm.that(pag.size, eq=c.DEFAULT_PAGE_SIZE)

    def test_query_validate_pagination_from_model(self) -> None:
        pag = m.Pagination(page=7, size=30)
        query = m.Query(pagination=pag)
        tm.that(query.pagination, is_=m.Pagination)
        pag = cast("m.Pagination", query.pagination)
        tm.that(pag.page, eq=7)
        tm.that(pag.size, eq=30)

    def test_query_validate_pagination_from_t_dict(self) -> None:
        parsed = m.Query.model_validate({
            "pagination": t.Dict(root={"page": 5, "size": 12}),
            "filters": {},
        })
        tm.that(parsed.pagination, is_=m.Pagination)
        pag = cast("m.Pagination", parsed.pagination)
        tm.that(pag.page, eq=5)
        tm.that(pag.size, eq=12)

    def test_query_resolve_pagination_wrapper(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Wrapper:
            class Pagination(m.Pagination):
                pass

            class Query(m.Query):
                pass

        Wrapper.Query.__module__ = "flext_core.models"
        Wrapper.Query.__qualname__ = "Wrapper.Query"
        mock_module: ModuleType = ModuleType("flext_core.models")
        setattr(mock_module, "Wrapper", Wrapper)
        monkeypatch.setitem(sys.modules, "flext_core.models", mock_module)
        tm.that(
            u.resolve_nested_model_class(
                module_name=Wrapper.Query.__module__,
                qualname=Wrapper.Query.__qualname__,
                models_module_name="flext_core.models",
                attribute_name="Pagination",
                fallback=m.Pagination,
            ),
            eq=Wrapper.Pagination,
        )

    def test_query_resolve_pagination_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Wrapper:
            class Query(m.Query):
                pass

        Wrapper.Query.__module__ = "flext_core.models"
        Wrapper.Query.__qualname__ = "Wrapper.Query"
        monkeypatch.setitem(
            sys.modules,
            "flext_core.models",
            ModuleType("flext_core.models"),
        )
        tm.that(
            u.resolve_nested_model_class(
                module_name=Wrapper.Query.__module__,
                qualname=Wrapper.Query.__qualname__,
                models_module_name="flext_core.models",
                attribute_name="Pagination",
                fallback=m.Pagination,
            ),
            eq=m.Pagination,
        )

    def test_query_resolve_pagination_deeper_nesting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Wrapper:
            class Inner:
                class Query(m.Query):
                    pass

        Wrapper.Inner.Query.__module__ = "flext_core.models"
        Wrapper.Inner.Query.__qualname__ = "Wrapper.Inner.Query"
        mock_module = ModuleType("flext_core.models")
        setattr(mock_module, "Wrapper", Wrapper)
        monkeypatch.setitem(sys.modules, "flext_core.models", mock_module)
        tm.that(
            u.resolve_nested_model_class(
                module_name=Wrapper.Inner.Query.__module__,
                qualname=Wrapper.Inner.Query.__qualname__,
                models_module_name="flext_core.models",
                attribute_name="Pagination",
                fallback=m.Pagination,
            ),
            eq=m.Pagination,
        )

    def test_query_filters_default_empty(self) -> None:
        query = m.Query()
        tm.that(query.filters.root, eq={})

    # ── Event ──────────────────────────────────────────────────

    def test_event_creation(self) -> None:
        evt = m.Event(event_type="user_created", aggregate_id="agg-1")
        tm.that(evt.message_type, eq="event")
        tm.that(evt.event_type, eq="user_created")
        tm.that(evt.aggregate_id, eq="agg-1")
        tm.that(evt.event_id, starts="evt_")
        tm.that(evt.tag, eq="event")

    def test_event_custom_data_and_metadata(self) -> None:
        evt = m.Event(
            event_type="order_placed",
            aggregate_id="order-99",
            data=t.Dict(root={"amount": 100}),
            metadata=t.Dict(root={"source": "api"}),
        )
        tm.that(evt.data.root["amount"], eq=100)
        tm.that(evt.metadata.root["source"], eq="api")

    def test_event_command_type_always_none(self) -> None:
        evt = m.Event(event_type="x", aggregate_id="y")
        tm.that(evt.command_type, none=True)

    def test_event_query_type_always_none(self) -> None:
        evt = m.Event(event_type="x", aggregate_id="y")
        tm.that(evt.query_type, none=True)

    def test_event_unique_ids(self) -> None:
        e1 = m.Event(event_type="a", aggregate_id="b")
        e2 = m.Event(event_type="a", aggregate_id="b")
        tm.that(e1.event_id, ne=e2.event_id)

    def test_event_rejects_empty_event_type(self) -> None:
        with pytest.raises(ValidationError):
            m.Event(event_type="", aggregate_id="agg-1")

    def test_event_rejects_empty_aggregate_id(self) -> None:
        with pytest.raises(ValidationError):
            m.Event(event_type="test", aggregate_id="")

    def test_event_rejects_missing_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            m.Event.model_validate({})

    def test_event_serialization_round_trip(self) -> None:
        evt = m.Event(
            event_type="item_deleted",
            aggregate_id="item-5",
            event_id="evt_fixed_id",
        )
        data = evt.model_dump()
        tm.that(data["message_type"], eq="event")
        tm.that(data["event_type"], eq="item_deleted")
        tm.that(data["aggregate_id"], eq="item-5")
        tm.that(data["event_id"], eq="evt_fixed_id")
        restored = m.Event.model_validate(data)
        tm.that(restored.event_type, eq=evt.event_type)
        tm.that(restored.event_id, eq=evt.event_id)

    def test_event_defaults_empty_data_and_metadata(self) -> None:
        evt = m.Event(event_type="a", aggregate_id="b")
        tm.that(evt.data.root, eq={})
        tm.that(evt.metadata.root, eq={})

    def test_event_message_type_frozen(self) -> None:
        with pytest.raises(ValidationError):
            m.Event.model_validate({
                "message_type": "command",
                "event_type": "x",
                "aggregate_id": "y",
            })

    # ── Handler ────────────────────────────────────────────────

    def test_handler_creation(self) -> None:
        handler = m.Handler(
            handler_id="h-1",
            handler_name="TestHandler",
            handler_type=c.HandlerType.COMMAND,
        )
        tm.that(handler.handler_id, eq="h-1")
        tm.that(handler.handler_name, eq="TestHandler")
        tm.that(handler.handler_type, eq=c.HandlerType.COMMAND)
        tm.that(handler.handler_mode, eq=c.HandlerType.COMMAND)
        tm.that(handler.command_timeout, eq=c.DEFAULT_MAX_COMMAND_RETRIES)
        tm.that(handler.max_command_retries, eq=c.DEFAULT_MAX_COMMAND_RETRIES)
        tm.that(handler.metadata, none=True)

    def test_handler_with_metadata(self) -> None:
        meta = m.Metadata(version="2.0.0", created_by="test")
        handler = m.Handler(
            handler_id="h-meta",
            handler_name="MetaHandler",
            metadata=meta,
        )
        tm.that(handler.metadata, none=False)
        if handler.metadata is not None:
            tm.that(handler.metadata.version, eq="2.0.0")

    @pytest.mark.parametrize(
        "handler_type",
        [c.HandlerType.COMMAND, c.HandlerType.QUERY, c.HandlerType.EVENT],
        ids=["command", "query", "event"],
    )
    def test_handler_accepts_handler_types(self, handler_type: c.HandlerType) -> None:
        handler = m.Handler(
            handler_id="h-param",
            handler_name="Param",
            handler_type=handler_type,
        )
        tm.that(handler.handler_type, eq=handler_type)

    def test_handler_rejects_missing_handler_name(self) -> None:
        with pytest.raises(ValidationError):
            m.Handler.model_validate({})

    def test_handler_serialization_round_trip(self) -> None:
        handler = m.Handler(
            handler_id="h-rt",
            handler_name="RoundTrip",
            handler_type=c.HandlerType.QUERY,
        )
        data = handler.model_dump()
        restored = m.Handler.model_validate(data)
        tm.that(restored.handler_id, eq="h-rt")
        tm.that(restored.handler_type, eq=c.HandlerType.QUERY)

    def test_handler_direct_custom_fields(self) -> None:
        handler = m.Handler(
            handler_id="custom-id",
            handler_name="MyHandler",
            command_timeout=30,
            max_command_retries=5,
        )
        tm.that(handler.handler_id, eq="custom-id")
        tm.that(handler.handler_name, eq="MyHandler")
        tm.that(handler.command_timeout, eq=30)
        tm.that(handler.max_command_retries, eq=5)

    def test_handler_query_mode_can_be_passed_directly(self) -> None:
        handler = m.Handler(
            handler_id="query-1",
            handler_name="QueryHandler",
            handler_type=c.HandlerType.QUERY,
            handler_mode=c.HandlerType.QUERY,
        )
        tm.that(handler.handler_type, eq=c.HandlerType.QUERY)
        tm.that(handler.handler_mode, eq=c.HANDLER_MODE_QUERY)

    def test_handler_model_validate_accepts_metadata_mapping(self) -> None:
        handler = m.Handler.model_validate({
            "handler_id": "merged-1",
            "handler_name": "Merged",
            "command_timeout": 60,
            "max_command_retries": 3,
            "metadata": {"version": "3.0.0"},
        })
        tm.that(handler.handler_name, eq="Merged")
        tm.that(handler.command_timeout, eq=60)
        tm.that(handler.max_command_retries, eq=3)
        tm.that(handler.metadata, none=False)
        if handler.metadata is not None:
            tm.that(handler.metadata.version, eq="3.0.0")

    # ── FlextMessage discriminated union ───────────────────────

    def test_message_union_command(self) -> None:
        adapter: TypeAdapter[m.FlextMessage] = TypeAdapter(m.FlextMessage)
        parsed = adapter.validate_python({
            "message_type": "command",
            "command_type": "run",
        })
        tm.that(parsed, is_=m.Command)
        tm.that(parsed.message_type, eq="command")

    def test_message_union_query(self) -> None:
        adapter: TypeAdapter[m.FlextMessage] = TypeAdapter(m.FlextMessage)
        parsed = adapter.validate_python({
            "message_type": "query",
        })
        tm.that(parsed, is_=m.Query)
        tm.that(parsed.message_type, eq="query")

    def test_message_union_event(self) -> None:
        adapter: TypeAdapter[m.FlextMessage] = TypeAdapter(m.FlextMessage)
        parsed = adapter.validate_python({
            "message_type": "event",
            "event_type": "happened",
            "aggregate_id": "agg-1",
        })
        tm.that(parsed, is_=m.Event)
        tm.that(parsed.message_type, eq="event")

    def test_message_union_invalid_type(self) -> None:
        adapter: TypeAdapter[m.FlextMessage] = TypeAdapter(m.FlextMessage)
        with pytest.raises(ValidationError):
            adapter.validate_python({"message_type": "invalid"})

    # ── parse_message raises NotImplementedError ───────────────

    def test_parse_message_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError):
            m.parse_message({"message_type": "command"})

    # ── Cross-type properties ──────────────────────────────────

    @pytest.mark.parametrize(
        ("model_factory", "expected_tag"),
        [
            (lambda: m.Command(), "command"),
            (lambda: m.Query(), "query"),
            (lambda: m.Event(event_type="x", aggregate_id="y"), "event"),
        ],
        ids=["command", "query", "event"],
    )
    def test_tag_class_var(
        self,
        model_factory: Callable[[], BaseModel],
        expected_tag: str,
    ) -> None:
        instance = model_factory()

    @pytest.mark.parametrize(
        ("model_factory", "has_command_type", "has_query_type", "has_event_type"),
        [
            (lambda: m.Command(), True, False, False),
            (lambda: m.Query(), False, False, False),
            (
                lambda: m.Event(event_type="x", aggregate_id="y"),
                False,
                False,
                True,
            ),
        ],
        ids=["command", "query", "event"],
    )
    def test_cross_type_properties(
        self,
        model_factory: Callable[[], BaseModel],
        has_command_type: bool,
        has_query_type: bool,
        has_event_type: bool,
    ) -> None:
        instance = model_factory()
        if has_command_type:
        else:
        if has_event_type:
        else:
        if has_query_type:
        else:
