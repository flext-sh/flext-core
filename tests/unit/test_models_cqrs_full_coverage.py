from __future__ import annotations

import sys
import importlib
from types import SimpleNamespace

from pydantic import TypeAdapter

from flext_core import c, m
from flext_core._models.cqrs import FlextMessage


def test_command_validator_and_pagination_limit() -> None:
    assert m.Cqrs.Command.validate_command(None) == "Command"
    assert m.Cqrs.Command.validate_command(123) == "123"
    page = m.Pagination(page=3, size=11)
    assert page.limit == 11


def test_query_resolve_pagination_wrapper_and_fallback(monkeypatch) -> None:
    class Wrapper:
        class Pagination(m.Pagination):
            pass

        class Query(m.Query):
            pass

    Wrapper.Query.__module__ = "flext_core.models"
    Wrapper.Query.__qualname__ = "Wrapper.Query"

    monkeypatch.setitem(
        sys.modules, "flext_core.models", SimpleNamespace(Wrapper=Wrapper)
    )
    assert Wrapper.Query._resolve_pagination_class() is Wrapper.Pagination

    monkeypatch.setitem(sys.modules, "flext_core.models", None)
    assert Wrapper.Query._resolve_pagination_class() is m.Pagination


def test_query_validate_pagination_dict_and_default() -> None:
    parsed = m.Query.model_validate({
        "pagination": {"page": "4", "size": "20"},
        "filters": {},
    })
    assert isinstance(parsed.pagination, m.Pagination)
    assert parsed.pagination.page == 4
    assert parsed.pagination.size == 20

    defaulted = m.Query.model_validate({"pagination": None, "filters": {}})
    assert isinstance(defaulted.pagination, m.Pagination)
    assert defaulted.pagination.page == c.Pagination.DEFAULT_PAGE_NUMBER


def test_handler_builder_fluent_methods() -> None:
    metadata = m.Metadata(attributes={"owner": "tests"})
    built = (
        m.Cqrs.Handler
        .Builder(c.Cqrs.HandlerType.QUERY)
        .with_id("handler-id")
        .with_name("handler-name")
        .with_timeout(33)
        .with_retries(7)
        .with_metadata(metadata)
        .merge_config(m.ConfigMap(root={"extra": "ok"}))
        .build()
    )
    assert built.handler_id == "handler-id"
    assert built.handler_name == "handler-name"
    assert built.command_timeout == 33
    assert built.max_command_retries == 7
    assert built.metadata is not None
    assert built.metadata.attributes["owner"] == "tests"


def test_cqrs_query_resolve_deeper_and_int_pagination(monkeypatch) -> None:
    class Wrapper:
        class Inner:
            class Query(m.Query):
                pass

    Wrapper.Inner.Query.__module__ = "flext_core.models"
    Wrapper.Inner.Query.__qualname__ = "Wrapper.Inner.Query"
    monkeypatch.setitem(
        sys.modules, "flext_core.models", SimpleNamespace(Wrapper=Wrapper)
    )
    assert Wrapper.Inner.Query._resolve_pagination_class() is m.Pagination

    parsed = m.Query.model_validate({
        "pagination": {"page": 2, "size": 10},
        "filters": {},
    })
    assert isinstance(parsed.pagination, m.Pagination)
    assert parsed.pagination.page == 2
    assert parsed.pagination.size == 10


def test_flext_message_discriminator_union_parsing() -> None:
    command = m.Cqrs.parse_message({"message_type": "command", "command_type": "sync"})
    assert isinstance(command, m.Command)
    assert command.message_type == "command"

    query = m.Cqrs.parse_message({"message_type": "query", "filters": {"k": "v"}})
    assert isinstance(query, m.Query)
    assert query.message_type == "query"

    event = m.Cqrs.parse_message({
        "message_type": "event",
        "event_type": "created",
        "aggregate_id": "agg-1",
        "data": {"x": 1},
    })
    assert isinstance(event, m.Cqrs.Event)
    assert event.message_type == "event"


def test_flext_message_type_alias_adapter() -> None:
    adapter: TypeAdapter[FlextMessage] = TypeAdapter(FlextMessage)
    parsed = adapter.validate_python({"message_type": "command", "command_type": "run"})
    assert isinstance(parsed, m.Command)
