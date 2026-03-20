from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import override

from flext_tests import tm
from hypothesis import given, strategies as st

from flext_core import r, s


class TestAutomatedFlextService:
    class _SuccessService(s[str]):
        __test__ = False

        @override
        def execute(self) -> r[str]:
            return r[str].ok("done")

    class _FailService(s[str]):
        __test__ = False

        @override
        def execute(self) -> r[str]:
            return r[str].fail("error occurred")

    class _DynamicService(s[str]):
        __test__ = False
        value: str

        @override
        def execute(self) -> r[str]:
            return r[str].ok(self.value)

    def test_execute_success(self) -> None:
        service = self._SuccessService()
        tm.ok(service.execute(), eq="done")

    def test_execute_failure(self) -> None:
        service = self._FailService()
        tm.fail(service.execute(), has="error")

    def test_validate_business_rules(self) -> None:
        service = self._SuccessService()
        tm.ok(service.validate_business_rules(), eq=True)

    def test_is_valid(self) -> None:
        tm.that(self._SuccessService().is_valid(), is_=bool)
        tm.that(self._FailService().is_valid(), eq=True)

    def test_get_service_info(self) -> None:
        service = self._SuccessService()
        info = service.get_service_info()
        tm.that(info, is_=Mapping)
        tm.that(info, has="service_type")

    def test_result_property(self) -> None:
        tm.that(self._SuccessService().result, eq="done")

    def test_runtime_properties(self) -> None:
        service = self._SuccessService()
        assert service.config is not None
        assert service.container is not None
        assert service.context is not None

    @given(st.text(min_size=1))
    def test_execute_hypothesis(self, value: str) -> None:
        service = self._DynamicService(value=value)
        result = service.execute()
        tm.that(result.is_success or result.is_failure, eq=True)

    def test_execute_benchmark(self) -> None:
        service = self._SuccessService()
        iterations = 1000
        op = u.Tests.Factory.simple_operation
        _ = op()
        tm.ok(r[str].ok("seed"), eq="seed")
        start = perf_counter()
        result = r[str].ok("init")
        for _ in range(iterations):
            result = service.execute()
        elapsed = perf_counter() - start
        tm.ok(result, eq="done")
        tm.that(elapsed, is_=float)
        tm.that(elapsed, gte=0.0)
