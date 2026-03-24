"""Golden-file example for FlextDecorators (d) public APIs."""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import override

from pydantic import BaseModel

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextExceptions,
    FlextRuntime,
    c,
    d,
    m,
    r,
    t,
    u,
)

from .shared import Examples


class Ex09FlextDecorators(Examples):
    """Exercise FlextDecorators public APIs."""

    _token_service_name: str
    _token_service_value: str
    _flaky_service_name: str
    _flaky_service_value: str

    @override
    def exercise(self) -> None:
        """Run all decorator demonstrations and record golden output."""
        self._setup_container()
        self._demo_deprecated()
        self._demo_inject()
        self._demo_factory()
        self._demo_railway()
        self._demo_retry()
        self._demo_timeout()
        self._demo_log_operation()
        self._demo_track_operation()
        self._demo_track_performance()
        self._demo_with_context()
        self._demo_with_correlation()
        self._demo_combined()

    def _demo_combined(self) -> None:
        """Exercise combined decorator in non-railway and railway modes."""
        self.section("combined")
        combined_standard_operation = self.rand_str(12)
        combined_railway_operation = self.rand_str(12)

        @d.combined(
            inject_deps={"service": self._token_service_name},
            operation_name=combined_standard_operation,
            track_perf=False,
            use_railway=False,
        )
        def combined_standard(*, service: str | None = None) -> str:
            """Use combined decorator without railway wrapping."""
            op_name = FlextContext.Request.get_operation_name()
            service_value = service if u.is_type(service, str) else "none"
            return f"{service_value}|{op_name}"

        @d.combined(
            inject_deps={"service": self._flaky_service_name},
            operation_name=combined_railway_operation,
            track_perf=True,
            use_railway=True,
            error_code="E_COMBINED",
        )
        def combined_railway(ok: bool, *, service: str | None = None) -> str:
            """Use combined decorator with railway wrapping and failure mapping."""
            if not ok:
                msg = self.rand_str(12)
                raise ValueError(msg)
            service_value = service if u.is_type(service, str) else "none"
            return f"{service_value}|{FlextContext.Request.get_operation_name()}"

        std_result = combined_standard()
        rail_ok = combined_railway(True)
        rail_fail = combined_railway(False)
        self.check(
            "combined.standard_matches",
            std_result == f"{self._token_service_value}|{combined_standard_operation}",
        )
        self.check("combined.railway.ok.is_success", rail_ok.is_success)
        self.check(
            "combined.railway.ok.value_matches",
            rail_ok.unwrap_or("none")
            == f"{self._flaky_service_value}|{combined_railway_operation}",
        )
        self.check("combined.railway.fail.is_failure", rail_fail.is_failure)
        self.check("combined.railway.fail.error_nonempty", bool(rail_fail.error))
        self.check(
            "combined.railway.fail.error_code",
            rail_fail.error_code == "E_COMBINED",
        )

    def _demo_deprecated(self) -> None:
        """Exercise deprecated decorator warning behavior."""
        self.section("deprecated")
        decorator_exists = hasattr(FlextDecorators, "deprecated")
        self.check("deprecated.exists", decorator_exists)
        if not decorator_exists:
            return
        deprecated_note = self.rand_str(12)
        source_value = self.rand_int(1, 1000)

        @d.deprecated(deprecated_note)
        def old_api(value: int) -> int:
            """Return incremented value to prove call still executes."""
            return value + 1

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", DeprecationWarning)
            result = old_api(source_value)
        self.check("deprecated.result_matches", result == source_value + 1)
        self.check("deprecated.warning_count", len(caught) == 1)
        self.check(
            "deprecated.warning_type",
            (caught[0].category.__name__ if caught else "none") == "DeprecationWarning",
        )
        self.check(
            "deprecated.warning_message",
            deprecated_note in (str(caught[0].message) if caught else "none"),
        )

    def _demo_factory(self) -> None:
        """Exercise factory decorator metadata with all parameter variations."""
        self.section("factory")
        factory_default_name = f"svc.{self.rand_str(6)}"
        factory_custom_name = f"svc.{self.rand_str(6)}"
        default_value = self.rand_str(10)
        custom_value = self.rand_str(10)

        class _FactoryPayload(m.Value):
            value: str

        @d.factory(name=factory_default_name)
        def factory_default(_: BaseModel) -> _FactoryPayload:
            """Factory function with default singleton and lazy values."""
            return _FactoryPayload(value=default_value)

        @d.factory(name=factory_custom_name, singleton=True, lazy=False)
        def factory_custom(_: BaseModel) -> _FactoryPayload:
            """Factory function with explicit singleton and lazy values."""
            return _FactoryPayload(value=custom_value)

        attr_name = c.FACTORY_ATTR
        default_cfg = getattr(factory_default, attr_name)
        custom_cfg = getattr(factory_custom, attr_name)
        self.check(
            "factory.default.name_matches",
            getattr(default_cfg, "name", "") == factory_default_name,
        )
        self.check("factory.default.singleton", getattr(default_cfg, "singleton", None))
        self.check("factory.default.lazy", getattr(default_cfg, "lazy", None))
        self.check(
            "factory.custom.name_matches",
            getattr(custom_cfg, "name", "") == factory_custom_name,
        )
        self.check("factory.custom.singleton", getattr(custom_cfg, "singleton", None))
        self.check("factory.custom.lazy", getattr(custom_cfg, "lazy", None))
        default_result = factory_default(t.ConfigMap(root={}))
        self.check(
            "factory.default.call_matches",
            default_result.model_dump().get("value") == default_value
            if default_result is not None
            else False,
        )
        custom_result = factory_custom(t.ConfigMap(root={}))
        self.check(
            "factory.custom.call_matches",
            custom_result.model_dump().get("value") == custom_value
            if custom_result is not None
            else False,
        )

    def _demo_inject(self) -> None:
        """Exercise inject decorator with container and override variations."""
        self.section("inject")

        @d.inject(service=self._token_service_name)
        def token_value(*, service: str | None = None) -> str:
            """Resolve token service from container."""
            if isinstance(service, str):
                return service
            return "none"

        override_service = self.rand_str(12)
        self.check(
            "inject.container_resolution_matches",
            token_value() == self._token_service_value,
        )
        self.check(
            "inject.kwarg_override_matches",
            token_value(service=override_service) == override_service,
        )
        missing_default_value = self.rand_str(8)
        missing_provided_value = self.rand_str(8)

        @d.inject(missing=f"svc.{self.rand_str(6)}")
        def missing_with_default(*, missing: str | None = None) -> str:
            """Keep running when dependency is not registered."""
            return missing_default_value if missing is None else missing_provided_value

        self.check(
            "inject.missing_dependency_default_matches",
            missing_with_default() == missing_default_value,
        )

    def _demo_log_operation(self) -> None:
        """Exercise log_operation with named/default operation and perf toggle."""
        self.section("log_operation")
        named_operation = self.rand_str(12)

        @d.log_operation(named_operation)
        def log_named() -> str:
            """Return operation name from context when explicitly set."""
            op_name = FlextContext.Request.get_operation_name()
            return op_name if op_name is not None else "none"

        @d.log_operation(track_perf=True)
        def log_default_perf() -> str:
            """Return operation name from context using default function name."""
            op_name = FlextContext.Request.get_operation_name()
            return op_name if op_name is not None else "none"

        self.check("log_operation.named_matches", log_named() == named_operation)
        self.check(
            "log_operation.default_track_perf",
            log_default_perf() == "log_default_perf",
        )

    def _demo_railway(self) -> None:
        """Exercise railway decorator success and failure mappings."""
        self.section("railway")

        @d.railway()
        def add(a: int, b: int) -> int:
            """Return plain value that should be wrapped in result.ok."""
            return a + b

        @d.railway(error_code="E_RAILWAY")
        def fail_railway() -> int:
            """Raise to trigger result.fail mapping."""
            msg = self.rand_str(12)
            raise ValueError(msg)

        left = self.rand_int(1, 500)
        right = self.rand_int(1, 500)
        ok_result = add(left, right)
        fail_result = fail_railway()
        alias_value = self.rand_int(1, 500)
        self.check("railway.alias_r.ok", r[int].ok(alias_value).value == alias_value)
        self.check("railway.ok.is_success", ok_result.is_success)
        self.check("railway.ok.value_matches", ok_result.unwrap_or(-1) == left + right)
        self.check("railway.fail.is_failure", fail_result.is_failure)
        self.check("railway.fail.error_nonempty", bool(fail_result.error))
        self.check("railway.fail.error_code", fail_result.error_code == "E_RAILWAY")

    def _demo_retry(self) -> None:
        """Exercise retry decorator with default, linear, and exponential paths."""
        self.section("retry")

        @d.retry()
        def retry_default() -> str:
            """Default retry configuration on immediate success."""
            return self.rand_str(10)

        linear_state = {"attempts": 0}

        @d.retry(max_attempts=3, delay_seconds=0.001, backoff_strategy="linear")
        def retry_linear() -> str:
            """Fail once then succeed with linear strategy."""
            linear_state["attempts"] += 1
            if linear_state["attempts"] < 2:
                msg = self.rand_str(10)
                raise RuntimeError(msg)
            return f"linear-{linear_state['attempts']}"

        exp_state = {"attempts": 0}

        @d.retry(max_attempts=2, delay_seconds=0.001, backoff_strategy="exponential")
        def retry_exponential() -> str:
            """Fail once then succeed with exponential strategy."""
            exp_state["attempts"] += 1
            if exp_state["attempts"] < 2:
                msg = self.rand_str(10)
                raise ValueError(msg)
            return f"exp-{exp_state['attempts']}"

        @d.retry(max_attempts=2, delay_seconds=0.001, error_code="E_RETRY")
        def retry_fails() -> str:
            """Always fail to prove raised error path."""
            msg = self.rand_str(12)
            raise RuntimeError(msg)

        self.check("retry.default.nonempty", bool(retry_default()))
        self.check("retry.linear.result", retry_linear() == "linear-2")
        self.check("retry.linear.attempts", linear_state["attempts"] == 2)
        self.check("retry.exponential.result", retry_exponential() == "exp-2")
        self.check("retry.exponential.attempts", exp_state["attempts"] == 2)
        try:
            retry_fails()
            self.check("retry.failure.raised", False)
        except RuntimeError as exc:
            self.check("retry.failure.raised", True)
            self.check("retry.failure.type", type(exc).__name__ == "RuntimeError")
            self.check("retry.failure.message_nonempty", bool(str(exc)))

    def _demo_timeout(self) -> None:
        """Exercise timeout decorator default and explicit timeout parameters."""
        self.section("timeout")

        @d.timeout()
        def timeout_default() -> str:
            """Default timeout path should pass quickly."""
            return self.rand_str(12)

        @d.timeout(timeout_seconds=5.0)
        def timeout_explicit_ok() -> str:
            """Explicit timeout path should pass quickly."""
            return self.rand_str(12)

        @d.timeout(timeout_seconds=0.0, error_code="E_TIMEOUT")
        def timeout_fail() -> str:
            """Force timeout by sleeping past a zero-second limit."""
            time.sleep(0.01)
            return self.rand_str(6)

        self.check("timeout.default.ok_nonempty", bool(timeout_default()))
        self.check("timeout.explicit.ok_nonempty", bool(timeout_explicit_ok()))
        try:
            timeout_fail()
            self.check("timeout.failure.raised", False)
        except FlextExceptions.TimeoutError as exc:
            self.check("timeout.failure.raised", True)
            self.check("timeout.failure.type", type(exc).__name__ == "TimeoutError")
            self.check("timeout.failure.error_code", exc.error_code == "E_TIMEOUT")

    def _demo_track_operation(self) -> None:
        """Exercise track_operation with correlation on and off."""
        self.section("track_operation")
        FlextContext.Utilities.clear_context()
        with_corr_operation = self.rand_str(12)

        @d.track_operation(with_corr_operation, track_correlation=True)
        def tracked_with_correlation() -> tuple[str | None, str | None]:
            """Return operation and correlation while inside decorator scope."""
            return (
                FlextContext.Request.get_operation_name(),
                FlextContext.Correlation.get_correlation_id(),
            )

        with_corr = tracked_with_correlation()
        FlextContext.Utilities.clear_context()

        @d.track_operation(track_correlation=False)
        def tracked_without_correlation() -> tuple[str | None, str | None]:
            """Return operation and correlation when correlation is not forced."""
            return (
                FlextContext.Request.get_operation_name(),
                FlextContext.Correlation.get_correlation_id(),
            )

        without_corr = tracked_without_correlation()
        self.check(
            "track_operation.with_corr.operation",
            with_corr[0] == with_corr_operation,
        )
        self.check("track_operation.with_corr.has_corr", with_corr[1] is not None)
        self.check(
            "track_operation.no_corr.operation",
            without_corr[0] == "tracked_without_correlation",
        )
        self.check("track_operation.no_corr.has_corr", without_corr[1] is not None)

    def _demo_track_performance(self) -> None:
        """Exercise track_performance with named and default operation names."""
        self.section("track_performance")
        perf_operation_name = self.rand_str(10)

        @d.log_operation(perf_operation_name)
        def perf_named(value: int) -> tuple[int, str | None]:
            """Return transformed value plus operation name from context."""
            return (value * 2, FlextContext.Request.get_operation_name())

        @d.log_operation()
        def perf_default() -> str | None:
            """Return default operation name from context."""
            return FlextContext.Request.get_operation_name()

        base_value = self.rand_int(1, 100)
        named_value, named_operation = perf_named(base_value)
        self.check("track_performance.named.value", named_value == base_value * 2)
        self.check(
            "track_performance.named.operation",
            named_operation == perf_operation_name,
        )
        self.check(
            "track_performance.default.operation",
            perf_default() == "perf_default",
        )

    def _demo_with_context(self) -> None:
        """Exercise with_context binding and None-value filtering."""
        self.section("with_context")
        FlextContext.Utilities.clear_context()
        tenant = self.rand_str(8)
        retries = self.rand_int(1, 10)
        enabled = self.rand_bool()

        @d.with_context(tenant=tenant, retries=retries, enabled=enabled, dropped=None)
        def read_bound_context() -> Mapping[str, t.Scalar | None]:
            """Read context values while decorator-managed context is active."""
            context_vars = dict(FlextRuntime.structlog().contextvars.get_contextvars())
            return {
                "tenant": context_vars.get("tenant"),
                "retries": context_vars.get("retries"),
                "enabled": context_vars.get("enabled"),
                "dropped": context_vars.get("dropped"),
            }

        inside = read_bound_context()
        after = dict(FlextRuntime.structlog().contextvars.get_contextvars())
        inside_dropped = inside.get("dropped")
        after_tenant_raw = after.get("tenant")
        after_tenant = (
            after_tenant_raw if u.is_type(after_tenant_raw, (str, int, bool)) else None
        )
        self.check("with_context.inside.tenant", inside.get("tenant") == tenant)
        self.check("with_context.inside.retries", inside.get("retries") == retries)
        self.check("with_context.inside.enabled", inside.get("enabled") == enabled)
        self.check(
            "with_context.inside.dropped",
            "None" if inside_dropped is None else inside_dropped,
        )
        self.check(
            "with_context.after.tenant",
            "None" if after_tenant is None else after_tenant,
        )

    def _demo_with_correlation(self) -> None:
        """Exercise with_correlation correlation-id creation behavior."""
        self.section("with_correlation")
        FlextContext.Utilities.clear_context()

        @d.with_correlation()
        def read_correlation() -> str | None:
            """Return correlation id ensured by decorator."""
            return FlextContext.Correlation.get_correlation_id()

        corr_id = read_correlation()
        self.check("with_correlation.created", corr_id is not None)
        self.check(
            "with_correlation.prefix",
            isinstance(corr_id, str) and corr_id.startswith("corr_"),
        )

    def _setup_container(self) -> FlextContainer:
        """Register services used by decorator examples."""
        container = FlextContainer.create()
        FlextContext.Utilities.clear_context()
        self._token_service_name = f"svc.{self.rand_str(6)}"
        self._token_service_value = self.rand_str(12)
        self._flaky_service_name = f"svc.{self.rand_str(6)}"
        self._flaky_service_value = self.rand_str(12)
        _ = container.register(self._token_service_name, self._token_service_value)
        _ = container.register(self._flaky_service_name, self._flaky_service_value)
        return container


if __name__ == "__main__":
    Ex09FlextDecorators(__file__).run()
