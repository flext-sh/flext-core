"""FlextLogger — exercises ALL public API methods with golden file validation."""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

from flext_core import FlextLogger, FlextRuntime, c, u

FlextRuntime.configure_structlog()

_RESULTS: list[str] = []


def _check(label: str, value: object) -> None:
    _RESULTS.append(f"{label}: {_ser(value)}")


def _section(name: str) -> None:
    if _RESULTS:
        _RESULTS.append("")
    _RESULTS.append(f"[{name}]")


def _ser(v: object) -> str:
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return repr(v)
    if u.is_list(v):
        return "[" + ", ".join(_ser(x) for x in v) + "]"
    if u.is_dict_like(v):
        pairs = ", ".join(
            f"{_ser(k)}: {_ser(val)}"
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        )
        return "{" + pairs + "}"
    if isinstance(v, type):
        return v.__name__
    return type(v).__name__


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


@dataclasses.dataclass
class _ContainerConfig:
    """Minimal container configuration stub."""

    log_level: str = "INFO"


class _ContainerStub:
    def __init__(self, log_level: str) -> None:
        self.config = _ContainerConfig(log_level)


def demo_factory_methods() -> None:
    """Exercise logger factory class methods."""
    _section("factory_methods")

    logger = FlextLogger.create_module_logger("examples.ex_03.factory")
    _check("create_module_logger.type", type(logger).__name__)

    raw = FlextLogger.get_logger("examples.ex_03.factory.raw")
    _check("get_logger.type", type(raw).__name__)

    wrapped = FlextLogger.create_bound_logger("examples.ex_03.factory.bound", raw)
    _check("create_bound_logger.type", type(wrapped).__name__)


def demo_global_context() -> None:
    """Exercise global context bind, unbind, and clear."""
    _section("global_context")

    logger = FlextLogger.create_module_logger("examples.ex_03.global")
    _check(
        "bind_global_context.ok",
        FlextLogger.bind_global_context(
            app_name="flext-core",
            correlation_id="g-001",
        ).is_success,
    )
    _check("global.info.ok", logger.info("global bound").is_success)
    _check(
        "unbind_global_context.ok",
        FlextLogger.unbind_global_context("correlation_id").is_success,
    )
    _check("clear_global_context.ok", FlextLogger.clear_global_context().is_success)


def demo_scoped_context() -> None:
    """Exercise scoped context management."""
    _section("scoped_context")

    logger = FlextLogger.create_module_logger("examples.ex_03.scope")
    application_scope = c.Context.SCOPE_APPLICATION
    request_scope = c.Context.SCOPE_REQUEST
    operation_scope = c.Context.SCOPE_OPERATION

    _check(
        "bind_context.application.ok",
        FlextLogger.bind_context(application_scope, app="core").is_success,
    )
    _check(
        "bind_context.request.ok",
        FlextLogger.bind_context(request_scope, request_id="req-1").is_success,
    )
    _check(
        "bind_context.operation.ok",
        FlextLogger.bind_context(operation_scope, operation="sync").is_success,
    )
    with FlextLogger.scoped_context(request_scope, tenant="acme"):
        _check(
            "scoped_context.info.ok", logger.info("inside scoped context").is_success
        )

    _check(
        "clear_scope.application.ok",
        FlextLogger.clear_scope(application_scope).is_success,
    )
    _check("clear_scope.request.ok", FlextLogger.clear_scope(request_scope).is_success)
    _check(
        "clear_scope.operation.ok", FlextLogger.clear_scope(operation_scope).is_success
    )


def demo_level_context() -> None:
    """Exercise level-specific context binding."""
    _section("level_context")

    logger = FlextLogger.create_module_logger("examples.ex_03.level")
    _check(
        "bind_context_for_level.ok",
        FlextLogger.bind_context_for_level("INFO", level_tag="l1").is_success,
    )
    _check("level.info.ok", logger.info("info with level context").is_success)
    _check(
        "unbind_context_for_level.ok",
        FlextLogger.unbind_context_for_level("INFO", "level_tag").is_success,
    )


def demo_container_integration() -> None:
    """Exercise container integration methods."""
    _section("container")

    container = _ContainerStub(log_level="DEBUG")
    logger = FlextLogger.for_container(container, level="DEBUG", worker="w-1")
    _check("for_container.type", type(logger).__name__)
    _check("for_container.debug.ok", logger.debug("for_container debug").is_success)
    with FlextLogger.with_container_context(container, level="INFO", feature="demo"):
        _check(
            "with_container_context.info.ok", logger.info("container scope").is_success
        )


def demo_instance_methods() -> None:
    """Exercise instance bind, unbind, logging, and exception methods."""
    _section("instance_methods")

    logger = FlextLogger.create_module_logger("examples.ex_03.instance")
    bound = logger.bind(component="demo")
    _check("bind.type", type(bound).__name__)

    renewed = bound.new(stage="new")
    _check("new.type", type(renewed).__name__)

    unbound = renewed.unbind("stage")
    _check("unbind.type", type(unbound).__name__)

    safe = unbound.unbind("missing", "component", safe=True)
    _check("try_unbind.type", type(safe).__name__)

    adapter = safe.with_result()
    _check("with_result.type", type(adapter).__name__)

    _check("trace.ok", safe.trace("trace value=%s", 1, key="t").is_success)
    _check("debug.ok", safe.debug("debug value=%s", 2, key="d").is_success)
    _check("info.ok", safe.info("info value=%s", 3, key="i").is_success)
    _check("warning.ok", safe.warning("warn value=%s", 4, key="w").is_success)
    _check("error.ok", safe.error("error value=%s", 6, key="e").is_success)
    _check("critical.ok", safe.critical("critical value=%s", 7, key="c").is_success)
    _check("log.ok", safe.log("INFO", "log value=%s", 8, key="l").is_success)

    try:
        boom_msg = "boom"
        raise ValueError(boom_msg)
    except ValueError as err:
        context_map = safe.build_exception_context(
            exception=err,
            exc_info=True,
            context={
                "source": "demo_instance_methods",
                "step": "build_exception_context",
            },
        )
        _check("build_exception_context.type", type(context_map).__name__)
        _check(
            "exception.ok",
            safe.exception(
                "exception path",
                exception=err,
                exc_info=True,
                source="demo_instance_methods",
            ).is_success,
        )


def demo_performance_tracker() -> None:
    """Exercise PerformanceTracker context manager."""
    _section("performance_tracker")

    logger = FlextLogger.create_module_logger("examples.ex_03.performance")
    with FlextLogger.PerformanceTracker(logger, "op.example"):
        _check("performance_tracker.body.ok", logger.info("within tracker").is_success)


def demo_result_adapter() -> None:
    """Exercise ResultAdapter wrapper methods."""
    _section("result_adapter")

    logger = FlextLogger.create_module_logger("examples.ex_03.adapter")
    adapter = logger.with_result()
    _check("adapter.name", adapter.name)
    _check("adapter.with_result.idempotent", adapter.with_result() is adapter)

    rebound = adapter.bind(adapter_key="v")
    _check("adapter.bind.type", type(rebound).__name__)

    _check("adapter.trace.ok", adapter.trace("adapter trace").is_success)
    _check("adapter.debug.ok", adapter.debug("adapter debug").is_success)
    _check("adapter.info.ok", adapter.info("adapter info").is_success)
    _check("adapter.warning.ok", adapter.warning("adapter warning").is_success)
    _check("adapter.error.ok", adapter.error("adapter error").is_success)
    _check("adapter.critical.ok", adapter.critical("adapter critical").is_success)

    try:
        adapter_msg = "adapter boom"
        raise RuntimeError(adapter_msg)
    except RuntimeError as err:
        _check(
            "adapter.exception.ok",
            adapter.exception(
                "adapter exception", exception=err, exc_info=True
            ).is_success,
        )


def main() -> None:
    """Run all logger demonstrations and verify golden file."""
    demo_factory_methods()
    demo_global_context()
    demo_scoped_context()
    demo_level_context()
    demo_container_integration()
    demo_instance_methods()
    demo_performance_tracker()
    demo_result_adapter()
    _verify()


if __name__ == "__main__":
    main()
