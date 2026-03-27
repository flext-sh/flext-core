"""FlextSettings API tests — merged from test_automated_settings.py."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from time import perf_counter

import pytest
from flext_tests import c as ftc, tf, tm, u
from hypothesis import given, settings, strategies as st
from pydantic_settings import BaseSettings

from flext_core import FlextSettings
from tests import c, t


class TestFlextSettingsCoverage:
    @pytest.fixture(autouse=True)
    def _reset_settings_state(self) -> Generator[None]:
        FlextSettings.reset_for_testing()
        yield
        FlextSettings.reset_for_testing()

    def test_get_global_and_apply_override(self) -> None:
        with tm.scope(config={"debug": False}):
            settings_obj = FlextSettings.get_global()
        tm.that(settings_obj, is_=FlextSettings)
        tm.that(settings_obj.apply_override("debug", True), eq=True)
        tm.that(settings_obj.debug, eq=True)
        tm.that(not settings_obj.apply_override("invalid_key", "x"), eq=True)

    def test_for_context_overrides(self) -> None:
        scoped = FlextSettings.for_context("ctx-1", debug=True, max_workers=99)
        tm.that(scoped.debug, eq=True)
        tm.that(scoped.max_workers, eq=99)

    def test_register_namespace_and_get_namespace(self) -> None:
        class DemoNamespace(BaseSettings):
            enabled: bool = True

        namespace = f"ns_{u.generate('ulid', length=6)}"
        FlextSettings.register_namespace(namespace, DemoNamespace)
        settings_obj = FlextSettings.get_global()
        ns_cfg = settings_obj.get_namespace(namespace, DemoNamespace)
        tm.that(ns_cfg.enabled, eq=True)

    def test_effective_log_level_property(self) -> None:
        settings_obj = FlextSettings.get_global(
            overrides={"debug": True, "trace": False},
        )
        tm.that(settings_obj.effective_log_level, eq=c.LogLevel.INFO)

    def test_reset_for_testing_creates_new_instance(self) -> None:
        first = FlextSettings.get_global()
        FlextSettings.reset_for_testing()
        second = FlextSettings.get_global()
        tm.that(first is not second, eq=True)

    def test_create_and_read_config_file(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        config = t.ConfigMap(root={"app_name": "flext", "debug": True, "port": 8080})
        config_path = files.create(
            config, "config.yaml", fmt=ftc.Tests.Files.Format.YAML
        )
        tm.that(config_path.exists(), eq=True)
        read_result = files.read(config_path, fmt=ftc.Tests.Files.Format.YAML)
        tm.ok(read_result)
        tm.that(read_result.value, is_=t.ConfigMap)
        if isinstance(read_result.value, t.ConfigMap):
            tm.that(str(read_result.value.root.get("app_name")), eq="flext")

    def test_create_and_read_json_config(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        payload = t.ConfigMap(
            root={"name": "flext-core", "workers": 4, "enabled": True},
        )
        config_path = files.create(
            payload, "config.json", fmt=ftc.Tests.Files.Format.JSON
        )
        tm.that(config_path.exists(), eq=True)
        read_result = files.read(config_path, fmt=ftc.Tests.Files.Format.JSON)
        tm.ok(read_result)
        tm.that(read_result.value, is_=t.ConfigMap)
        if isinstance(read_result.value, t.ConfigMap):
            workers = read_result.value.root.get("workers")
            tm.that(workers, is_=int)
            if isinstance(workers, int):
                tm.that(workers, eq=4)

    def test_compare_identical_files(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        first = files.create(
            t.ConfigMap(root={"x": 1}), "a.json", fmt=ftc.Tests.Files.Format.JSON
        )
        second = files.create(
            t.ConfigMap(root={"x": 1}), "b.json", fmt=ftc.Tests.Files.Format.JSON
        )
        result = files.compare(first, second)
        tm.ok(result)
        tm.that(result.value, eq=True)

    @given(
        pair=st.one_of(
            st.tuples(st.just("debug"), st.booleans()),
            st.tuples(st.just("trace"), st.booleans()),
            st.tuples(st.just("max_workers"), st.integers(min_value=1, max_value=100)),
            st.tuples(
                st.just("log_level"),
                st.sampled_from([
                    c.LogLevel.DEBUG,
                    c.LogLevel.INFO,
                    c.LogLevel.WARNING,
                    c.LogLevel.ERROR,
                ]),
            ),
            st.tuples(st.just("invalid_key"), st.text(min_size=1, max_size=10)),
        ),
    )
    @settings(max_examples=50)
    def test_apply_override_returns_bool_property(
        self,
        pair: tuple[str, bool | int | str | c.LogLevel],
    ) -> None:
        """Property: apply_override always returns bool."""
        key, value = pair
        FlextSettings.reset_for_testing()
        settings_obj = FlextSettings.get_global()
        if key == "trace" and value is True:
            tm.that(settings_obj.apply_override("debug", True), eq=True)
        outcome = settings_obj.apply_override(key, value)
        tm.that(outcome, is_=bool)

    @pytest.mark.performance
    def test_apply_override_benchmark(self) -> None:
        settings_obj = FlextSettings.get_global()
        keys = ["debug", "trace", "max_workers"]
        formatter = u.Tests.Factory.format_operation
        tm.that(callable(formatter), eq=True)
        start = perf_counter()
        for idx in range(400):
            key = keys[idx % len(keys)]
            value = True if key in {"debug", "trace"} else len(str(formatter("w", idx)))
            tm.that(settings_obj.apply_override(key, value), eq=True)
            _ = settings_obj.effective_log_level
        tm.that(perf_counter() - start, gte=0.0)


__all__ = ["TestFlextSettingsCoverage"]
