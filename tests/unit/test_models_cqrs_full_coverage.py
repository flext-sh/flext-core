"""Tests for CQRS models full coverage."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest
from pydantic import TypeAdapter

from tests import c, m


def test_command_pagination_limit() -> None:
    page = m.Pagination(page=3, size=11)
    assert page.limit == 11


def test_query_resolve_pagination_wrapper_and_fallback(
    monkeypatch: pytest.MonkeyPatch,
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
    assert Wrapper.Query._resolve_pagination_class() is Wrapper.Pagination
    monkeypatch.setitem(
        sys.modules,
        "flext_core.models",
        ModuleType("flext_core.models"),
    )
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
    assert defaulted.pagination.page == c.DEFAULT_PAGE_NUMBER


def test_handler_builder_fluent_methods() -> None:
    handler = m.Handler(
        handler_type=c.HandlerType.QUERY,
        handler_id="h-1",
        handler_name="handler",
    )
    assert handler.handler_type == c.HandlerType.QUERY
    assert handler.handler_id == "h-1"


def test_cqrs_query_resolve_deeper_and_int_pagination(
    monkeypatch: pytest.MonkeyPatch,
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
    assert Wrapper.Inner.Query._resolve_pagination_class() is m.Pagination
    parsed = m.Query.model_validate({
        "pagination": {"page": 2, "size": 10},
        "filters": {},
    })
    assert isinstance(parsed.pagination, m.Pagination)
    assert parsed.pagination.page == 2
    assert parsed.pagination.size == 10


def test_flext_message_type_alias_adapter() -> None:
    adapter: TypeAdapter[m.MessageUnion] = TypeAdapter(m.MessageUnion)
    parsed = adapter.validate_python({"message_type": "command", "command_type": "run"})
    assert type(parsed).__name__ == "CommandMessage"
    assert parsed.message_type == "command"
