"""Real FlextContainer API tests using flext_tests infrastructure."""

from __future__ import annotations

import shutil
from collections.abc import Generator
from time import perf_counter

import pytest
from flext_tests import tk, tm
from hypothesis import given, settings, strategies as st

from flext_core import FlextContainer, FlextContext

_DOCKER_AVAILABLE = shutil.which("docker") is not None


class TestAutomatedFlextContainer:
    @pytest.fixture(autouse=True)
    def _reset_container_state(self) -> Generator[None]:
        FlextContainer.reset_for_testing()
        yield
        FlextContainer.reset_for_testing()

    def test_register_and_get_service(self) -> None:
        container = FlextContainer.create()

        def my_factory() -> str:
            return "hello"

        _ = container.register("my_svc", my_factory, kind="factory")
        tm.ok(container.get("my_svc"), eq="hello")

    def test_get_unregistered_fails(self) -> None:
        container = FlextContainer.create()
        tm.fail(container.get("nonexistent"), has="not found")

    def test_has_service_and_list_services(self) -> None:
        container = FlextContainer.create()
        names = ["svc_one", "svc_two"]

        def first_factory() -> str:
            return names[0]

        def second_factory() -> str:
            return names[1]

        _ = container.register(names[0], first_factory, kind="factory")
        _ = container.register(names[1], second_factory, kind="factory")
        tm.that(container.has_service("svc_one"), eq=True)
        tm.that(container.has_service("svc_two"), eq=True)
        tm.that(container.has_service("svc_missing"), eq=False)
        listed = container.list_services()
        tm.that(listed, has=["svc_one", "svc_two"])

    def test_unregister_and_clear_all(self) -> None:
        container = FlextContainer.create()

        def temp_factory() -> int:
            return 99

        _ = container.register("temp", temp_factory, kind="factory")
        tm.ok(container.unregister("temp"), eq=True)
        tm.fail(container.get("temp"), has="not found")

        def factory_a() -> int:
            return 1

        def factory_b() -> int:
            return 2

        _ = container.register("a", factory_a, kind="factory")
        _ = container.register("b", factory_b, kind="factory")
        container.clear_all()
        tm.that(container.list_services(), length=0)

    def test_scoped_and_configure(self) -> None:
        container = FlextContainer.create()
        _ = {"max_services": 10}
        config_override = {"max_services": 10}
        _ = container.configure(config_override)
        tm.that(container.get_config()["max_services"], eq=10)
        scoped = container.scoped(
            context=FlextContext.create(),
            subproject="unit",
            services={"scoped_service": "scoped-value"},
        )
        tm.that(scoped.has_service("scoped_service"), eq=True)
        tm.ok(scoped.get("scoped_service"), eq="scoped-value")
        tm.ok(scoped.context.get("subproject"), eq="unit")

    def test_reset_for_testing_creates_new_singleton(self) -> None:
        first = FlextContainer.create()
        FlextContainer.reset_for_testing()
        second = FlextContainer.create()
        tm.that(first is second, eq=False)

    @given(
        name=st.text(
            min_size=1,
            max_size=30,
            alphabet=st.characters(min_codepoint=48, max_codepoint=122),
        )
    )
    @settings(max_examples=50)
    def test_register_get_roundtrip(self, name: str) -> None:
        container = FlextContainer.create()
        sanitized = "".join(ch for ch in name if ch.isalnum()) or "svc"

        def dynamic_factory() -> str:
            return sanitized

        _ = container.register(sanitized, dynamic_factory, kind="factory")
        tm.ok(container.get(sanitized), eq=sanitized)

    @pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker not available")
    @pytest.mark.integration
    def test_docker_container_status(self) -> None:
        docker = tk()
        status = docker.get_container_status("nonexistent_container")
        tm.fail(status, has="not found")

    @pytest.mark.performance
    def test_register_get_benchmark(self) -> None:
        container = FlextContainer.create()
        op = u.Tests.Factory.add_operation
        tm.that(callable(op), eq=True)

        def bench_factory() -> int:
            return 42

        _ = container.register("bench_svc", bench_factory, kind="factory")
        start = perf_counter()
        for _ in range(500):
            tm.ok(container.get("bench_svc"), eq=42)
        elapsed = perf_counter() - start
        tm.that(elapsed, gte=0.0)
