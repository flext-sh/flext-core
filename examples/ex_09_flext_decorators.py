"""Golden-file example for FlextDecorators (d) public APIs."""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

from flext_core import (
    FlextContainer,
    FlextContext,
    FlextDecorators,
    FlextExceptions,
    FlextRuntime,
    c,
    d,
    r,
    t,
    u,
)

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(_coerce(value))}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _coerce(value: object) -> t.ContainerValue:
    if value is None:
        return "None"
    if isinstance(value, (str, int, float, bool, Path)):
        return value
    if isinstance(value, tuple):
        return "tuple"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return repr(value)


def _ser(v: t.ContainerValue) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list) and u.is_list(v):
        list_value: list[t.ContainerValue] = v
        return "[" + ", ".join(_ser(item) for item in list_value) + "]"
    if isinstance(v, dict) and u.is_dict_like(v):
        dict_value: dict[str, t.ContainerValue] = v
        pairs = ", ".join(
            f"{_ser(key)}: {_ser(value)}"
            for key, value in sorted(dict_value.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    return repr(v)


def _verify() -> None:
    actual = "\n".join(_RESULTS).strip() + "\n"
    me = Path(__file__)
    expected_path = me.with_suffix(".expected")
    n = sum(1 for line in _RESULTS if ": " in line and not line.startswith("["))
    if expected_path.exists():
        expected = expected_path.read_text(encoding="utf-8")
        if actual == expected:
            sys.stdout.write(f"PASS: {me.stem} ({n} checks)\n")
        else:
            actual_path = me.with_suffix(".actual")
            actual_path.write_text(actual, encoding="utf-8")
            sys.stdout.write(
                f"FAIL: {me.stem} — diff {expected_path.name} {actual_path.name}\n"
            )
            sys.exit(1)
    else:
        expected_path.write_text(actual, encoding="utf-8")
        sys.stdout.write(f"GENERATED: {expected_path.name} ({n} checks)\n")


def _setup_container() -> FlextContainer:
    """Register services used by decorator examples."""
    container = FlextContainer.create()
    FlextContext.Utilities.clear_context()
    _ = container.register("ex09.token", "token-from-container")
    _ = container.register("ex09.flaky", "pong")
    return container


def demo_deprecated() -> None:
    """Exercise deprecated decorator warning behavior."""
    _section("deprecated")

    decorator_exists = hasattr(FlextDecorators, "deprecated")
    _check("deprecated.exists", decorator_exists)

    if not decorator_exists:
        return

    @d.deprecated("use new_api instead")
    def old_api(value: int) -> int:
        """Return an incremented value to prove call still executes."""
        return value + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        result = old_api(10)

    _check("deprecated.result", result)
    _check("deprecated.warning_count", len(caught))
    _check(
        "deprecated.warning_type",
        caught[0].category.__name__ if caught else "none",
    )
    _check(
        "deprecated.warning_message",
        str(caught[0].message) if caught else "none",
    )


def demo_inject() -> None:
    """Exercise inject decorator with container and override variations."""
    _section("inject")

    @d.inject(service="ex09.token")
    def token_value(*, service: str | None = None) -> str:
        """Resolve token service from container."""
        return service if isinstance(service, str) else "none"

    override_service = "override-token"
    _check("inject.container_resolution", token_value())
    _check("inject.kwarg_override", token_value(service=override_service))

    @d.inject(missing="ex09.missing")
    def missing_with_default(*, missing: object | None = None) -> str:
        """Keep running when dependency is not registered."""
        return "missing-default" if missing is None else "missing-provided"

    _check("inject.missing_dependency_default", missing_with_default())


def demo_factory() -> None:
    """Exercise factory decorator metadata with all parameter variations."""
    _section("factory")

    @d.factory(name="ex09.factory.default")
    def factory_default(_: object) -> str:
        """Factory function with default singleton/lazy values."""
        return "default"

    @d.factory(name="ex09.factory.custom", singleton=True, lazy=False)
    def factory_custom(_: object) -> str:
        """Factory function with explicit singleton/lazy values."""
        return "custom"

    attr_name = c.Discovery.FACTORY_ATTR
    default_cfg = getattr(factory_default, attr_name)
    custom_cfg = getattr(factory_custom, attr_name)

    _check("factory.default.name", getattr(default_cfg, "name", ""))
    _check("factory.default.singleton", getattr(default_cfg, "singleton", None))
    _check("factory.default.lazy", getattr(default_cfg, "lazy", None))
    _check("factory.custom.name", getattr(custom_cfg, "name", ""))
    _check("factory.custom.singleton", getattr(custom_cfg, "singleton", None))
    _check("factory.custom.lazy", getattr(custom_cfg, "lazy", None))
    _check("factory.default.call", factory_default("arg"))
    _check("factory.custom.call", factory_custom("arg"))


def demo_railway() -> None:
    """Exercise railway decorator success and failure mappings."""
    _section("railway")

    @d.railway()
    def add(a: int, b: int) -> int:
        """Return a plain value that should be wrapped in result.ok."""
        return a + b

    @d.railway(error_code="E_RAILWAY")
    def fail_railway() -> int:
        """Raise to trigger result.fail mapping."""
        msg = "railway-failure"
        raise ValueError(msg)

    ok_result = add(2, 3)
    fail_result = fail_railway()

    _check("railway.alias_r.ok", r[int].ok(9).unwrap_or(-1))
    _check("railway.ok.is_success", ok_result.is_success)
    _check("railway.ok.value", ok_result.unwrap_or(-1))
    _check("railway.fail.is_failure", fail_result.is_failure)
    _check("railway.fail.error", fail_result.error)
    _check("railway.fail.error_code", fail_result.error_code)


def demo_retry() -> None:
    """Exercise retry decorator defaults plus linear/exponential strategies."""
    _section("retry")

    @d.retry()
    def retry_default() -> str:
        """Default retry configuration on immediate success."""
        return "default-ok"

    linear_state = {"attempts": 0}

    @d.retry(max_attempts=3, delay_seconds=0.001, backoff_strategy="linear")
    def retry_linear() -> str:
        """Fail once then succeed with linear strategy."""
        linear_state["attempts"] += 1
        if linear_state["attempts"] < 2:
            msg = "linear-retry"
            raise RuntimeError(msg)
        return f"linear-{linear_state['attempts']}"

    exp_state = {"attempts": 0}

    @d.retry(max_attempts=2, delay_seconds=0.001, backoff_strategy="exponential")
    def retry_exponential() -> str:
        """Fail once then succeed with exponential strategy."""
        exp_state["attempts"] += 1
        if exp_state["attempts"] < 2:
            msg = "exp-retry"
            raise ValueError(msg)
        return f"exp-{exp_state['attempts']}"

    @d.retry(max_attempts=2, delay_seconds=0.001, error_code="E_RETRY")
    def retry_fails() -> str:
        """Always fail to prove raised error path."""
        msg = "retry-exhausted"
        raise RuntimeError(msg)

    _check("retry.default", retry_default())
    _check("retry.linear.result", retry_linear())
    _check("retry.linear.attempts", linear_state["attempts"])
    _check("retry.exponential.result", retry_exponential())
    _check("retry.exponential.attempts", exp_state["attempts"])

    try:
        retry_fails()
        _check("retry.failure.raised", False)
    except RuntimeError as exc:
        _check("retry.failure.raised", True)
        _check("retry.failure.type", type(exc).__name__)
        _check("retry.failure.message", str(exc))


def demo_timeout() -> None:
    """Exercise timeout decorator default and explicit timeout parameters."""
    _section("timeout")

    @d.timeout()
    def timeout_default() -> str:
        """Default timeout path should pass quickly."""
        return "default-timeout-ok"

    @d.timeout(timeout_seconds=5.0)
    def timeout_explicit_ok() -> str:
        """Explicit timeout path should pass quickly."""
        return "explicit-timeout-ok"

    @d.timeout(timeout_seconds=0.0, error_code="E_TIMEOUT")
    def timeout_fail() -> str:
        """Force timeout by sleeping past a zero-second limit."""
        time.sleep(0.01)
        return "late"

    _check("timeout.default.ok", timeout_default())
    _check("timeout.explicit.ok", timeout_explicit_ok())

    try:
        timeout_fail()
        _check("timeout.failure.raised", False)
    except FlextExceptions.TimeoutError as exc:
        _check("timeout.failure.raised", True)
        _check("timeout.failure.type", type(exc).__name__)
        _check("timeout.failure.error_code", exc.error_code)


def demo_log_operation() -> None:
    """Exercise log_operation with named/default operation and perf toggle."""
    _section("log_operation")

    @d.log_operation("named_log_operation")
    def log_named() -> str:
        """Return operation name from context when explicitly set."""
        op_name = FlextContext.Request.get_operation_name()
        return op_name if op_name is not None else "none"

    @d.log_operation(track_perf=True)
    def log_default_perf() -> str:
        """Return operation name from context using default function name."""
        op_name = FlextContext.Request.get_operation_name()
        return op_name if op_name is not None else "none"

    _check("log_operation.named", log_named())
    _check("log_operation.default_track_perf", log_default_perf())


def demo_track_operation() -> None:
    """Exercise track_operation with correlation on and off."""
    _section("track_operation")

    FlextContext.Utilities.clear_context()

    @d.track_operation("tracked_with_corr", track_correlation=True)
    def tracked_with_correlation() -> tuple[str | None, str | None]:
        """Return operation/correlation while inside decorator scope."""
        return (
            FlextContext.Request.get_operation_name(),
            FlextContext.Correlation.get_correlation_id(),
        )

    with_corr = tracked_with_correlation()

    FlextContext.Utilities.clear_context()

    @d.track_operation(track_correlation=False)
    def tracked_without_correlation() -> tuple[str | None, str | None]:
        """Return operation/correlation when correlation is not forced."""
        return (
            FlextContext.Request.get_operation_name(),
            FlextContext.Correlation.get_correlation_id(),
        )

    without_corr = tracked_without_correlation()

    _check("track_operation.with_corr.operation", with_corr[0])
    _check("track_operation.with_corr.has_corr", with_corr[1] is not None)
    _check("track_operation.no_corr.operation", without_corr[0])
    _check("track_operation.no_corr.has_corr", without_corr[1] is not None)


def demo_track_performance() -> None:
    """Exercise track_performance with named and default operation names."""
    _section("track_performance")

    @d.track_performance("perf_named")
    def perf_named(value: int) -> tuple[int, str | None]:
        """Return transformed value plus operation name from context."""
        return value * 2, FlextContext.Request.get_operation_name()

    @d.track_performance()
    def perf_default() -> str | None:
        """Return default operation name from context."""
        return FlextContext.Request.get_operation_name()

    named_value, named_operation = perf_named(4)
    _check("track_performance.named.value", named_value)
    _check("track_performance.named.operation", named_operation)
    _check("track_performance.default.operation", perf_default())


def demo_with_context() -> None:
    """Exercise with_context binding/unbinding and None-value filtering."""
    _section("with_context")

    FlextContext.Utilities.clear_context()

    @d.with_context(tenant="tenant-1", retries=2, enabled=True, dropped=None)
    def read_bound_context() -> dict[str, object | None]:
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

    _check("with_context.inside.tenant", inside.get("tenant"))
    _check("with_context.inside.retries", inside.get("retries"))
    _check("with_context.inside.enabled", inside.get("enabled"))
    _check("with_context.inside.dropped", inside.get("dropped"))
    _check("with_context.after.tenant", after.get("tenant"))


def demo_with_correlation() -> None:
    """Exercise with_correlation correlation-id creation behavior."""
    _section("with_correlation")

    FlextContext.Utilities.clear_context()

    @d.with_correlation()
    def read_correlation() -> str | None:
        """Return correlation id ensured by decorator."""
        return FlextContext.Correlation.get_correlation_id()

    corr_id = read_correlation()
    _check("with_correlation.created", corr_id is not None)
    _check(
        "with_correlation.prefix",
        isinstance(corr_id, str) and corr_id.startswith("corr_"),
    )


def demo_combined() -> None:
    """Exercise combined decorator in non-railway and railway modes."""
    _section("combined")

    @d.combined(
        inject_deps={"service": "ex09.token"},
        operation_name="combined_standard",
        track_perf=False,
        use_railway=False,
    )
    def combined_standard(*, service: str | None = None) -> str:
        """Use combined decorator without railway wrapping."""
        op_name = FlextContext.Request.get_operation_name()
        service_value = service if isinstance(service, str) else "none"
        return f"{service_value}|{op_name}"

    @d.combined(
        inject_deps={"service": "ex09.flaky"},
        operation_name="combined_railway",
        track_perf=True,
        use_railway=True,
        error_code="E_COMBINED",
    )
    def combined_railway(ok: bool, *, service: str | None = None) -> str:
        """Use combined decorator with railway wrapping and failure mapping."""
        if not ok:
            msg = "combined-failure"
            raise ValueError(msg)
        service_value = service if isinstance(service, str) else "none"
        return f"{service_value}|{FlextContext.Request.get_operation_name()}"

    std_result = combined_standard()
    rail_ok = combined_railway(True)
    rail_fail = combined_railway(False)

    _check("combined.standard", std_result)
    _check("combined.railway.ok.is_success", rail_ok.is_success)
    _check("combined.railway.ok.value", rail_ok.unwrap_or("none"))
    _check("combined.railway.fail.is_failure", rail_fail.is_failure)
    _check("combined.railway.fail.error", rail_fail.error)
    _check("combined.railway.fail.error_code", rail_fail.error_code)


def main() -> None:
    """Run all decorator demonstrations and verify golden output."""
    _ = _setup_container()
    demo_deprecated()
    demo_inject()
    demo_factory()
    demo_railway()
    demo_retry()
    demo_timeout()
    demo_log_operation()
    demo_track_operation()
    demo_track_performance()
    demo_with_context()
    demo_with_correlation()
    demo_combined()
    _verify()


if __name__ == "__main__":
    main()
