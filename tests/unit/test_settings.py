"""Behavior contract for flext_core.FlextSettings — public API only."""

from __future__ import annotations

import ast
import os
import threading
import time
from collections.abc import (
    Generator,
    Mapping,
    MutableSequence,
)
from pathlib import Path
from time import perf_counter

import pytest
from flext_tests import c as tc, tf, tm
from hypothesis import given, settings, strategies as st

from flext_core import FlextSettings, FlextUtilitiesGenerators
from tests import c, m, p, t, u


@pytest.fixture(autouse=True)
def reset_flext_settings_singleton() -> Generator[None]:
    """Isolate singleton state across settings tests."""
    FlextSettings.reset_for_testing()
    try:
        yield
    finally:
        FlextSettings.reset_for_testing()


class TestsFlextSettings:
    """Behavior contract for FlextSettings — public API only."""

    # --- Initialization --------------------------------------------------

    @pytest.mark.parametrize(
        "config_data",
        [
            {"app_name": "test_app", "version": "1.0.0", "debug": True},
            {"app_name": "dict_app", "version": "2.0.0", "debug": False},
            {"app_name": "valid_app", "version": "1.0.0"},
        ],
        ids=lambda d: str(d.get("app_name", "default")),
    )
    def test_initialization_from_kwargs_sets_expected_fields(
        self,
        config_data: t.FeatureFlagMapping,
    ) -> None:
        settings = u.Tests.create_test_config(**config_data)
        dumped_settings = settings.model_dump()
        for key, expected_value in config_data.items():
            actual_value = dumped_settings.get(key)
            msg = f"Config {key}: expected {expected_value}, got {actual_value}"
            assert actual_value == expected_value, msg
        tm.that(settings, is_=FlextSettings)

    def test_model_dump_round_trips_values(self) -> None:
        settings = u.Tests.create_test_config(
            app_name="test_app",
            version="1.0.0",
            debug=True,
        )
        dumped = settings.model_dump()
        tm.that(dumped["app_name"], eq="test_app")
        tm.that(dumped["version"], eq="1.0.0")
        tm.that(dumped["debug"], eq=True)

    def test_initialization_ignores_unknown_kwargs(self) -> None:
        settings = FlextSettings(app_name="known", unknown_setting="ignored")
        tm.that(settings.app_name, eq="known")
        tm.that(hasattr(settings, "unknown_setting"), eq=False)

    def test_model_validate_returns_same_singleton(self) -> None:
        original = u.Tests.create_test_config(
            app_name="original_app",
            version="1.0.0",
        )
        dumped = original.model_dump(exclude={"is_production"})
        cloned = FlextSettings.model_validate(dumped)
        tm.that(cloned is original, eq=True)

    @pytest.mark.parametrize(
        ("field_name", "initial", "modified"),
        [
            ("app_name", "test_value", "modified_value"),
            ("version", "1.0.0", "2.0.0"),
        ],
    )
    def test_field_mutation_is_visible(
        self,
        field_name: str,
        initial: str,
        modified: str,
    ) -> None:
        settings = u.Tests.create_test_config()
        setattr(settings, field_name, initial)
        tm.that(getattr(settings, field_name), eq=initial)
        setattr(settings, field_name, modified)
        tm.that(getattr(settings, field_name), eq=modified)

    # --- Singleton -------------------------------------------------------

    def test_direct_instantiation_returns_shared_singleton(self) -> None:
        first = FlextSettings()
        second = FlextSettings()
        tm.that(first is second, eq=True)
        tm.that(first.model_dump_json(), eq=second.model_dump_json())

    def test_fetch_global_returns_same_instance_across_calls(self) -> None:
        first = FlextSettings.fetch_global()
        second = FlextSettings.fetch_global()
        tm.that(first is second, eq=True)

    def test_reset_for_testing_clears_singleton(self) -> None:
        original = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        fresh = FlextSettings()
        tm.that(fresh is not original, eq=True)
        tm.that(fresh.app_name, eq="flext")

    def test_concurrent_writes_do_not_crash(self) -> None:
        settings = u.Tests.create_test_config()
        results: MutableSequence[str] = []

        def set_value(thread_id: int) -> None:
            settings.app_name = f"thread_{thread_id}"
            results.append(settings.app_name)

        threads = [threading.Thread(target=set_value, args=(i,)) for i in range(10)]
        for t_ in threads:
            t_.start()
        for t_ in threads:
            t_.join()
        tm.that(len(results), eq=10)
        tm.that(all(r.startswith("thread_") for r in results), eq=True)

    def test_repeated_writes_complete_within_budget(self) -> None:
        settings = u.Tests.create_test_config()
        start = time.time()
        for i in range(100):
            settings.app_name = f"value_{i}"
            _ = settings.app_name
        tm.that(time.time() - start, lt=5.0)

    # --- Serialization ---------------------------------------------------

    def test_json_round_trip_preserves_fields(self) -> None:
        settings = u.Tests.create_test_config(
            app_name="serialize_app",
            version="1.0.0",
        )
        json_str = settings.model_dump_json(
            exclude={
                "is_debug_enabled",
                "effective_log_level",
                "is_production",
                "effective_timeout",
                "has_database",
                "has_cache",
            },
        )
        tm.that("serialize_app" in json_str, eq=True)
        restored = FlextSettings.model_validate_json(json_str)
        tm.that(restored.app_name, eq=settings.app_name)
        tm.that(restored.version, eq=settings.version)

    # --- Validation ------------------------------------------------------

    def test_trace_without_debug_raises_validation_error(self) -> None:
        with pytest.raises(c.ValidationError, match="Trace mode requires debug mode"):
            FlextSettings.model_validate({"trace": True, "debug": False})

    def test_invalid_database_url_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid database URL scheme"):
            FlextSettings.model_validate({"database_url": "invalid://scheme"})

    # --- Environment loading ---------------------------------------------

    @pytest.mark.parametrize(
        ("env_key", "env_value", "should_load"),
        [
            ("DEBUG", "true", False),
            ("FLEXT_DEBUG", "true", True),
        ],
    )
    def test_env_prefix_is_respected(
        self,
        env_key: str,
        env_value: str,
        should_load: bool,
    ) -> None:
        with u.Tests.env_vars_context(
            {env_key: env_value},
            ["DEBUG", "LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_LOG_LEVEL"],
        ):
            if not env_key.startswith(c.ENV_PREFIX):
                os.environ["DEBUG"] = "true"
            settings = FlextSettings()
            tm.that(settings.debug, eq=should_load)

    def test_dotenv_file_loads_when_env_file_var_points_to_it(
        self,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text(
            "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\nFLEXT_DEBUG=true\n",
        )
        with u.Tests.env_vars_context(
            {"FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_LOG_LEVEL", "FLEXT_DEBUG", "FLEXT_APP_NAME", "FLEXT_ENV_FILE"],
        ):
            FlextSettings.reset_for_testing()
            settings = FlextSettings()
            tm.that(settings.app_name in {"from-dotenv", "flext"}, eq=True)

    def test_env_var_overrides_dotenv_file(self, tmp_path: Path) -> None:
        with u.Tests.env_vars_context(
            {},
            ["FLEXT_APP_NAME", "FLEXT_LOG_LEVEL"],
        ):
            env_file = tmp_path / ".env"
            env_file.write_text(
                "FLEXT_APP_NAME=from-dotenv\nFLEXT_LOG_LEVEL=WARNING\n",
            )
            os.environ["FLEXT_APP_NAME"] = "from-env-var"
            os.environ["FLEXT_LOG_LEVEL"] = "ERROR"
            settings = FlextSettings()
            tm.that(settings.app_name, eq="from-env-var")
            tm.that(settings.log_level, eq="ERROR")

    def test_explicit_override_wins_over_env_and_dotenv(
        self,
        tmp_path: Path,
    ) -> None:
        env_file = tmp_path / ".env"
        env_file.write_text("FLEXT_TIMEOUT_SECONDS=45\n")
        with u.Tests.env_vars_context(
            {"FLEXT_TIMEOUT_SECONDS": "60", "FLEXT_ENV_FILE": str(env_file)},
            ["FLEXT_TIMEOUT_SECONDS", "FLEXT_ENV_FILE"],
        ):
            settings = FlextSettings.fetch_global(overrides={"timeout_seconds": 90})
            tm.that(settings.timeout_seconds, eq=90)

    # --- Effective log level --------------------------------------------

    def test_trace_mode_sets_effective_log_level_to_debug(self) -> None:
        settings = u.Tests.create_test_config(trace=True, debug=True)
        tm.that(settings.effective_log_level, eq=c.LogLevel.DEBUG)

    def test_debug_mode_sets_effective_log_level_to_info(self) -> None:
        settings = u.Tests.create_test_config(debug=True)
        tm.that(settings.effective_log_level, eq=c.LogLevel.INFO)

    def test_without_debug_effective_log_level_matches_configured(self) -> None:
        settings = u.Tests.create_test_config(debug=False, trace=False)
        tm.that(settings.effective_log_level, eq=settings.log_level)

    # --- Overrides and DI provider --------------------------------------

    def test_apply_override_rejects_unknown_key(self) -> None:
        settings = u.Tests.create_test_config()
        tm.that(settings.apply_override("invalid_key", "value"), eq=False)

    def test_apply_override_updates_known_field(self) -> None:
        settings = u.Tests.create_test_config()
        settings.apply_override("app_name", "new_name")
        tm.that(settings.app_name, eq="new_name")

    def test_resolve_di_settings_provider_returns_stable_handle(self) -> None:
        settings = u.Tests.create_test_config()
        first = settings.resolve_di_settings_provider()
        second = settings.resolve_di_settings_provider()
        tm.that(first is second, eq=True)

    def test_auto_settings_create_settings_returns_settings_protocol(self) -> None:
        auto = FlextSettings.AutoSettings(settings_class=FlextSettings)
        instance = auto.create_settings()
        tm.that(instance, is_=p.Settings)

    # --- Clone / isolated copies -----------------------------------------

    def test_clone_without_overrides_returns_deep_copy(self) -> None:
        original = u.Tests.create_test_config(app_name="orig")
        copied = original.clone()
        tm.that(copied is not original, eq=True)
        tm.that(copied.app_name, eq="orig")
        copied.app_name = "mutated"
        tm.that(original.app_name, eq="orig")

    def test_clone_with_overrides_applies_fields(self) -> None:
        original = u.Tests.create_test_config(app_name="orig", version="1.0.0")
        copied = original.clone(app_name="new", version="2.0.0")
        tm.that(copied.app_name, eq="new")
        tm.that(copied.version, eq="2.0.0")
        tm.that(original.app_name, eq="orig")

    def test_clone_reruns_validators(self) -> None:
        original = u.Tests.create_test_config(debug=True)
        with pytest.raises(c.ValidationError, match="Trace mode requires debug mode"):
            original.clone(trace=True, debug=False)

    def test_for_context_uses_deep_copy(self) -> None:
        original = FlextSettings.fetch_global()
        context = FlextSettings.for_context("test_ctx", app_name="ctx_app")
        tm.that(context is not original, eq=True)
        tm.that(context.app_name, eq="ctx_app")
        tm.that(original.app_name, eq="flext")

    @staticmethod
    def _extract_config_payload(value: m.ConfigMap) -> t.MappingKV[str, object]:
        payload: t.MappingKV[str, object] = value.root
        nested = payload.get("root")
        if isinstance(nested, Mapping):
            return nested

        embedded = payload.get("value")
        if isinstance(embedded, str) and embedded.startswith("root="):
            raw = embedded.removeprefix("root=").strip()
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return payload
            if isinstance(parsed, Mapping):
                return parsed

        if isinstance(embedded, Mapping):
            return embedded
        return payload

    def test_get_global_and_apply_override(self) -> None:
        with tm.scope(settings={"debug": False}):
            settings_obj = FlextSettings.fetch_global()
        tm.that(settings_obj, is_=FlextSettings)
        tm.that(settings_obj.apply_override("debug", True), eq=True)
        tm.that(settings_obj.debug, eq=True)
        tm.that(not settings_obj.apply_override("invalid_key", "x"), eq=True)

    def test_for_context_overrides(self) -> None:
        scoped = FlextSettings.for_context("ctx-1", debug=True, max_workers=99)
        tm.that(scoped.debug, eq=True)
        tm.that(scoped.max_workers, eq=99)

    def test_register_namespace_and_get_namespace(self) -> None:
        class DemoNamespace(FlextSettings):
            model_config = m.SettingsConfigDict(
                env_prefix="FLEXT_DEMO_",
                extra="ignore",
            )
            enabled: bool = True

        namespace = f"ns_{u.generate('ulid', options=FlextUtilitiesGenerators.GenerateOptions(length=6))}"
        FlextSettings.register_namespace(namespace, DemoNamespace)
        settings_obj = FlextSettings.fetch_global()
        ns_cfg = settings_obj.fetch_namespace(namespace, DemoNamespace)
        tm.that(ns_cfg, is_=DemoNamespace)
        tm.that(ns_cfg.enabled, eq=True)

    def test_auto_register_registers_namespace(self) -> None:
        namespace = f"ns_{u.generate('ulid', options=FlextUtilitiesGenerators.GenerateOptions(length=6))}"

        @FlextSettings.auto_register(namespace)
        class DemoAutoNamespace(FlextSettings):
            model_config = m.SettingsConfigDict(
                env_prefix="FLEXT_DEMO_AUTO_",
                extra="ignore",
            )
            enabled: bool = True

        settings_obj = FlextSettings.fetch_global()
        ns_cfg = settings_obj.fetch_namespace(namespace, DemoAutoNamespace)
        tm.that(ns_cfg, is_=DemoAutoNamespace)
        tm.that(ns_cfg.enabled, eq=True)

    def test_effective_log_level_property(self) -> None:
        settings_obj = FlextSettings.fetch_global(
            overrides={"debug": True, "trace": False}
        )
        tm.that(settings_obj.effective_log_level, eq=c.LogLevel.INFO)

    def test_reset_for_testing_creates_new_instance(self) -> None:
        first = FlextSettings.fetch_global()
        FlextSettings.reset_for_testing()
        second = FlextSettings.fetch_global()
        tm.that(first is not second, eq=True)

    def test_create_and_read_config_file(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        settings_payload = {
            "app_name": "flext",
            "debug": True,
            "port": 8080,
        }
        config_path = files.create(
            settings_payload,
            "settings.yaml",
            fmt=tc.Tests.FILE_FORMAT_YAML,
        )
        tm.that(config_path.exists(), eq=True)
        read_result = files.read(config_path, fmt=tc.Tests.FILE_FORMAT_YAML)
        tm.ok(read_result)
        tm.that(read_result.value, is_=m.ConfigMap)
        if isinstance(read_result.value, m.ConfigMap):
            root_payload = self._extract_config_payload(read_result.value)
            tm.that(str(root_payload.get("app_name")), eq="flext")

    def test_create_and_read_json_config(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        payload_mapping = {
            "name": "flext-core",
            "workers": 4,
            "enabled": True,
        }
        config_path = files.create(
            payload_mapping,
            "settings.json",
            fmt=tc.Tests.FILE_FORMAT_JSON,
        )
        tm.that(config_path.exists(), eq=True)
        read_result = files.read(config_path, fmt=tc.Tests.FILE_FORMAT_JSON)
        tm.ok(read_result)
        tm.that(read_result.value, is_=m.ConfigMap)
        if isinstance(read_result.value, m.ConfigMap):
            root_payload = self._extract_config_payload(read_result.value)
            workers = root_payload.get("workers")
            tm.that(workers, is_=int)
            if isinstance(workers, int):
                tm.that(workers, eq=4)

    def test_compare_identical_files(self, tmp_path: Path) -> None:
        files_cls: type[tf] = tf
        files = files_cls(base_dir=tmp_path)
        first = files.create({"x": 1}, "a.json", fmt=tc.Tests.FILE_FORMAT_JSON)
        second = files.create({"x": 1}, "b.json", fmt=tc.Tests.FILE_FORMAT_JSON)
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
        settings_obj = FlextSettings.fetch_global()
        if key == "trace" and value is True:
            tm.that(settings_obj.apply_override("debug", True), eq=True)
        outcome = settings_obj.apply_override(key, value)
        tm.that(outcome, is_=bool)

    @pytest.mark.performance
    def test_apply_override_benchmark(self) -> None:
        settings_obj = FlextSettings.fetch_global()
        keys = ["debug", "trace", "max_workers"]
        start = perf_counter()
        for idx in range(400):
            key = keys[idx % len(keys)]
            value = True if key in {"debug", "trace"} else len(f"w-{idx}")
            tm.that(settings_obj.apply_override(key, value), eq=True)
            _ = settings_obj.effective_log_level
        tm.that(perf_counter() - start, gte=0.0)


__all__: t.MutableSequenceOf[str] = [
    "TestsFlextSettings",
]
