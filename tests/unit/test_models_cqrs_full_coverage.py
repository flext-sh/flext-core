"""Tests for CQRS models full coverage."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from tests import c, m


class TestsFlextCoreModelsCqrs:
    def test_command_pagination_limit(self) -> None:
        page = m.Pagination(page=3, size=11)
        assert page.limit == 11

    def test_query_resolve_pagination_with_custom_override(self) -> None:
        # Test that Query subclasses can override Pagination via class attribute
        class CustomQuery(m.Query):
            class Pagination(m.Pagination):
                pass

        wrapped = CustomQuery.model_validate({
            "pagination": {"page": 1, "size": 9},
            "filters": {},
        })
        # Should use the custom Pagination from the Query subclass
        assert type(wrapped.pagination) is CustomQuery.Pagination
        assert wrapped.pagination.page == 1
        assert wrapped.pagination.size == 9

    def test_query_validate_pagination_dict_and_default(self) -> None:
        parsed = m.Query.model_validate({
            "pagination": {"page": "4", "size": "20"},
            "filters": {},
        })
        assert isinstance(parsed.pagination, m.Pagination)
        assert parsed.pagination.page == 4
        assert parsed.pagination.size == 20
        defaulted = m.Query.model_validate({"pagination": None, "filters": {}})
        assert isinstance(defaulted.pagination, m.Pagination)
        assert defaulted.pagination.page == c.DEFAULT_RETRY_DELAY_SECONDS

    def test_handler_builder_fluent_methods(self) -> None:
        handler = m.Handler(
            handler_type=c.HandlerType.QUERY,
            handler_id="h-1",
            handler_name="handler",
        )
        assert handler.handler_type == c.HandlerType.QUERY
        assert handler.handler_id == "h-1"

    def test_cqrs_query_resolve_deeper_and_int_pagination(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:

        class Wrapper:
            class Inner:
                class Query(m.Query):
                    pass

        Wrapper.Inner.Query.__module__ = "flext_core"
        Wrapper.Inner.Query.__qualname__ = "Wrapper.Inner.Query"
        mock_module = ModuleType("flext_core")
        setattr(mock_module, "Wrapper", Wrapper)
        monkeypatch.setitem(sys.modules, "flext_core", mock_module)
        wrapped = Wrapper.Inner.Query.model_validate({
            "pagination": {"page": 2, "size": 10},
            "filters": {},
        })
        assert isinstance(wrapped.pagination, m.Pagination)
        parsed = m.Query.model_validate({
            "pagination": {"page": 2, "size": 10},
            "filters": {},
        })
        assert isinstance(parsed.pagination, m.Pagination)
        assert parsed.pagination.page == 2
        assert parsed.pagination.size == 10

    def test_flext_message_type_alias_adapter(self) -> None:
        adapter = m.TypeAdapter(m.FlextMessage.__value__)
        parsed = adapter.validate_python({
            "message_type": "command",
            "command_type": "run",
        })
        assert type(parsed).__name__ == "Command"
        assert parsed.message_type == "command"
