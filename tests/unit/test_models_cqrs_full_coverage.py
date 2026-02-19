from __future__ import annotations

import sys
import importlib
from types import SimpleNamespace

core = importlib.import_module("flext_core")
c = core.c
m = core.m


def test_command_validator_and_pagination_limit() -> None:
    assert m.Cqrs.Command.validate_command(None) == "Command"
    assert m.Cqrs.Command.validate_command(123) == "123"
    page = m.Cqrs.Pagination(page=3, size=11)
    assert page.limit == 11


def test_query_resolve_pagination_wrapper_and_fallback(monkeypatch) -> None:
    class Wrapper:
        class Pagination(m.Cqrs.Pagination):
            pass

        class Query(m.Cqrs.Query):
            pass

    Wrapper.Query.__module__ = "flext_core.models"
    Wrapper.Query.__qualname__ = "Wrapper.Query"

    monkeypatch.setitem(
        sys.modules, "flext_core.models", SimpleNamespace(Wrapper=Wrapper)
    )
    assert Wrapper.Query._resolve_pagination_class() is Wrapper.Pagination

    monkeypatch.setitem(sys.modules, "flext_core.models", None)
    assert Wrapper.Query._resolve_pagination_class() is m.Cqrs.Pagination


def test_query_validate_pagination_dict_and_default() -> None:
    parsed = m.Cqrs.Query.model_validate({
        "pagination": {"page": "4", "size": "20"},
        "filters": {},
    })
    assert isinstance(parsed.pagination, m.Cqrs.Pagination)
    assert parsed.pagination.page == 4
    assert parsed.pagination.size == 20

    defaulted = m.Cqrs.Query.model_validate({"pagination": None, "filters": {}})
    assert isinstance(defaulted.pagination, m.Cqrs.Pagination)
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
            class Query(m.Cqrs.Query):
                pass

    Wrapper.Inner.Query.__module__ = "flext_core.models"
    Wrapper.Inner.Query.__qualname__ = "Wrapper.Inner.Query"
    monkeypatch.setitem(
        sys.modules, "flext_core.models", SimpleNamespace(Wrapper=Wrapper)
    )
    assert Wrapper.Inner.Query._resolve_pagination_class() is m.Cqrs.Pagination

    parsed = m.Cqrs.Query.model_validate({
        "pagination": {"page": 2, "size": 10},
        "filters": {},
    })
    assert parsed.pagination.page == 2
    assert parsed.pagination.size == 10
