"""Typed stage registration through the real public logging boundary.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/unit/test_logging_stages
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import asyncio
import fcntl
import io
import logging
import multiprocessing
import os
import select
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import structlog
from pydantic import ValidationError

from tests import c, m, p, t, tm, u

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Never


class TestsFlextLoggingStages:
    """Observable contracts of typed event stages and terminal rendering."""

    @staticmethod
    def _make_observe(observed: list[p.LoggingEvent]) -> p.LoggingStage:
        """Capture the event before the terminal renderer consumes its fields.

        Returns:
            The typed stage preserving a snapshot of each observed event.

        """

        def observe(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name
            observed.append(dict(event_dict))
            return event_dict

        return observe

    @staticmethod
    def test_closed_async_stream_preserves_native_failure() -> None:
        """A failed sink terminates without logging back into its own writer."""
        consumer = (
            "import contextlib, io, logging, sys, structlog\n"
            "from flext_core import m, u\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=True, log_level=logging.INFO))\n"
            "stream = io.StringIO()\n"
            "stream.close()\n"
            "with contextlib.redirect_stderr(stream):\n"
            "    structlog.get_logger().info('closed stream')\n"
        )
        executed = tm.ok(u.Cli.run_raw((sys.executable, "-c", consumer)))
        tm.fail(u.Cli.process_succeeded(executed.outcome))
        tm.fail(executed.outcome.timed_out)
        tm.that(executed.outcome.forwarded_signal, eq=None)
        tm.that(executed.stderr, has="ValueError: I/O operation on closed file")

    @staticmethod
    def test_async_emission_drains_before_reconfiguration_and_process_exit() -> None:
        """Accepted public events reach the native stream before shutdown."""
        events = (
            " first accepted event ",
            "\tsecond accepted event\n",
            "last accepted event",
        )
        consumer = (
            "import logging, sys\n"
            "from flext_core import m, u\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=True, console_renderer=False, log_level=logging.INFO))\n"
            "logger = u.create_module_logger('tests.logging.native.consumer')\n"
            "for event in sys.argv[1:]:\n"
            "    logger.info(event)\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=False, console_renderer=False, log_level=logging.INFO))\n"
        )
        executed = tm.ok(
            u.Cli.run_raw((sys.executable, "-c", consumer, *events)),
        )
        tm.ok(u.Cli.process_succeeded(executed.outcome))
        tm.fail(executed.outcome.timed_out)
        emitted = tuple(
            u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(line)))["event"]
            for line in executed.stderr.splitlines()
        )
        tm.that(emitted, eq=events)
        tm.that(executed.stdout, empty=True)

    @staticmethod
    def test_failed_async_sink_rethrows_the_same_public_exception() -> None:
        """Subsequent public writes preserve the first native stream exception."""
        consumer = (
            "import contextlib, io, logging, sys\n"
            "from flext_core import m, u\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=True, log_level=logging.INFO))\n"
            "logger = u.create_module_logger('tests.logging.native.failure')\n"
            "stream = io.StringIO()\n"
            "stream.close()\n"
            "previous_stderr = sys.stderr\n"
            "sys.stderr = stream\n"
            "try:\n"
            "    logger.info('first failed event')\n"
            "except ValueError as first:\n"
            "    try:\n"
            "        logger.info('second failed event')\n"
            "    except ValueError as second:\n"
            "        if second is not first:\n"
            "            raise AssertionError('native failure identity changed')\n"
            "        sys.stderr = previous_stderr\n"
            "        raise first\n"
            "    raise AssertionError('second event did not fail')\n"
            "raise AssertionError('closed native stream did not fail')\n"
        )
        executed = tm.ok(u.Cli.run_raw((sys.executable, "-c", consumer)))
        tm.fail(u.Cli.process_succeeded(executed.outcome))
        tm.fail(executed.outcome.timed_out)
        tm.that(executed.outcome.forwarded_signal, eq=None)
        tm.that(executed.stderr, has="ValueError: I/O operation on closed file")
        tm.that(executed.stderr, lacks="AssertionError")

    @pytest.fixture(autouse=True)
    def restore_logging_options(self) -> Generator[None]:
        """Restore the actual captured configuration after each registration."""
        original_settings, original_options = u.logging_settings(), u.logging_options()
        try:
            selected = u.resolve_logging_settings(
                log_level=c.LogLevel.DEBUG, trace=False
            )
            u.configure_structlog(
                settings=m.StructlogOptions(async_logging=False),
                runtime_settings=selected,
            )
            yield
        finally:
            u.configure_structlog(
                settings=original_options, runtime_settings=original_settings
            )

    def test_ordered_stages_preserve_nested_values_and_render(self) -> None:
        """Stages share an event in order and the final renderer emits it."""
        stream = io.StringIO()
        observed: list[p.LoggingEvent] = []

        def first(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name
            event_dict["stage_order"] = ["first"]
            observed.append(dict(event_dict))
            return event_dict

        def second(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name
            tm.that(event_dict["stage_order"], eq=["first"])
            event_dict["stage_order"] = ["first", "second"]
            observed.append(dict(event_dict))
            return event_dict

        options = m.StructlogOptions(
            processing_stages=(first, second),
            console_renderer=False,
            logger_factory=structlog.PrintLoggerFactory(file=stream),
        )
        u.configure_structlog(settings=options)
        logger = u.create_module_logger("tests.logging.stages.ordered")
        payload: t.JsonDict = {"items": [1, {"enabled": True}], "empty": None}

        failure = ValueError("active exception context")
        try:
            raise failure
        except ValueError:
            result = logger.info("ordered stages", payload=payload, exc_info=True)

        tm.ok(result.success)
        tm.that(len(observed), eq=2)
        tm.that(observed[1]["payload"], eq=payload)
        tm.that(observed[1]["exception_type"], eq="ValueError")
        tm.ok('"stage_order": ["first", "second"]' in stream.getvalue())

    @pytest.mark.parametrize("entrypoint", ["info", "exception", "trace"])
    def test_first_stage_failure_escapes_unchanged(self, entrypoint: str) -> None:
        """Failure stops the chain and preserves the original exception."""
        stream = io.StringIO()
        visited: list[str] = []
        failure = ValueError("typed stage failed")

        def fail_first(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name, event_dict
            visited.append("first")
            raise failure

        def later(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name
            visited.append("later")
            return event_dict

        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                processing_stages=(fail_first, later),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        logger = u.create_module_logger(f"tests.logging.stages.failure.{entrypoint}")
        u.apply_log_level(log_level=c.LogLevel.DEBUG, trace=True)

        match entrypoint:
            case "exception":
                emit = logger.exception
            case "trace":
                emit = logger.trace
            case _:
                emit = logger.info
        with pytest.raises(ValueError, match="typed stage failed") as caught:
            _ = emit("stage failure")

        tm.ok(caught.value is failure)
        tm.that(visited, eq=["first"])
        tm.that(stream.getvalue(), eq="")
        frames = traceback.extract_tb(caught.value.__traceback__)
        tm.that(frames[-1].name, eq="fail_first")

    def test_exception_context_reaches_stage(self) -> None:
        """Native SDK stages receive causal metadata and the original exception."""
        observed: list[p.LoggingEvent] = []
        stream = io.StringIO()

        observe = self._make_observe(observed)

        u.configure_structlog(
            settings=m.StructlogOptions(
                processing_stages=(observe,),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        logger = u.create_module_logger("tests.logging.stages.exception")
        error = OSError("storage unavailable")
        try:
            raise error
        except OSError as exc:
            original_traceback = exc.__traceback__
            result = logger.exception(
                "operation failed",
                exception=exc,
                operation="read",
            )

        tm.ok(result.success)
        tm.that(observed[0]["exception_type"], eq=type(error).__name__)
        tm.that(observed[0]["exception_message"], eq=str(error))
        info = tm.not_none(u.resolve_logging_exception_info(observed[0]["exc_info"]))
        tm.ok(info[1] is error)
        tm.ok(info[2] is original_traceback)
        tm.ok("operation failed" in stream.getvalue())

    def test_invalid_registration_is_rejected(self) -> None:
        """The options boundary rejects a noncallable stage."""
        with pytest.raises(ValidationError):
            _ = m.StructlogOptions.model_validate({"processing_stages": [1]})

    @pytest.mark.parametrize("saved_tuple", [False, True])
    def test_native_exception_admission_keeps_ordinary_fields_json(
        self, *, saved_tuple: bool
    ) -> None:
        """Native causes retain identity while ordinary fields use canonical JSON."""
        stream = io.StringIO()
        observed: list[p.LoggingEvent] = []

        observe = self._make_observe(observed)

        u.configure_structlog(
            settings=m.StructlogOptions(
                console_renderer=False,
                processing_stages=(observe,),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            )
        )
        original = ValueError("saved native logging cause")
        try:
            raise original
        except ValueError:
            saved: t.LoggingExceptionInfo = (
                type(original),
                original,
                original.__traceback__,
            )
        payload = m.ConfigMap(root={"items": [1, {"enabled": True}]})
        source = Path(__file__)
        tm.ok(
            u.fetch_logger("native.exception.admission").error(
                "native exception with JSON fields",
                exc_info=saved if saved_tuple else original,
                payload=payload,
                sequence=(1, "two", False, None),
                source=source,
            )
        )
        info = tm.not_none(u.resolve_logging_exception_info(observed[0]["exc_info"]))
        tm.ok(info[0] is saved[0])
        tm.ok(info[1] is saved[1])
        tm.ok(info[2] is saved[2])
        rendered = u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(stream.getvalue())))
        tm.that(rendered["payload"], eq=payload.model_dump(mode="json"))
        tm.that(rendered["sequence"], eq=[1, "two", False, None])
        tm.that(rendered["source"], eq=str(source))
        tm.fail("exc_info" in rendered)
        tm.that(rendered["exception_message"], eq=str(original))

    def test_exception_triple_is_not_an_ordinary_json_field(self) -> None:
        """An unsupported exception tuple fails admission before any output."""
        stream = io.StringIO()
        observed: list[p.LoggingRecord] = []
        u.configure_structlog(
            settings=m.StructlogOptions(
                console_renderer=False,
                observers=(observed.append,),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            )
        )
        original = ValueError("not an ordinary JSON value")
        saved: t.LoggingExceptionInfo = type(original), original, None
        with pytest.raises(m.ValidationError):
            _ = u.fetch_logger("native.exception.field").info(
                "invalid JSON field", payload=saved
            )
        tm.that(observed, eq=[])
        tm.that(stream.getvalue(), eq="")

    def test_invalid_callable_signature_raises_at_emission(self) -> None:
        """An untyped registration cannot hide its native invocation error."""
        stream = io.StringIO()

        def invalid_stage() -> p.LoggingEvent:
            return {}

        options = m.StructlogOptions.model_validate({
            "processing_stages": [invalid_stage],
            "logger_factory": structlog.PrintLoggerFactory(file=stream),
        })
        u.configure_structlog(settings=options)
        logger = u.create_module_logger("tests.logging.stages.invalid")

        with pytest.raises(TypeError):
            _ = logger.info("invalid stage signature")

        tm.that(stream.getvalue(), eq="")

    def test_explicit_registration_updates_cached_logger_chain(self) -> None:
        """Cached and fresh public loggers consume the elected processing chain."""
        stream = io.StringIO()
        observed: list[str] = []
        initial = m.StructlogOptions(
            cache_logger_on_first_use=True,
            logger_factory=structlog.PrintLoggerFactory(file=stream),
        )
        u.configure_structlog(settings=initial)
        cached = u.create_module_logger("tests.logging.stages.cached")
        _ = cached.info("cache initialized")

        def observe(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: p.LoggingEvent,
        ) -> p.LoggingEvent:
            _ = logger, method_name
            observed.append(str(event_dict["event"]))
            return event_dict

        replacement = m.StructlogOptions(
            cache_logger_on_first_use=initial.cache_logger_on_first_use,
            logger_factory=initial.logger_factory,
            processing_stages=(observe,),
        )
        u.configure_structlog(settings=replacement)
        _ = cached.info("cached chain")
        fresh = u.create_module_logger("tests.logging.stages.fresh")
        result = fresh.info("replacement chain")

        tm.ok(result.success)
        tm.that(observed, eq=["cached chain", "replacement chain"])
        tm.ok("cached chain" in stream.getvalue())
        tm.ok("replacement chain" in stream.getvalue())

    @pytest.mark.parametrize("initial_async", [True, False])
    def test_explicit_synchronous_factory_replaces_previous_factory(
        self, *, initial_async: bool
    ) -> None:
        """Fresh synchronous loggers replace asynchronous and custom outputs."""
        old_stream = io.StringIO()
        first_options = m.StructlogOptions(
            async_logging=initial_async,
            logger_factory=(
                None if initial_async else structlog.PrintLoggerFactory(file=old_stream)
            ),
        )
        u.configure_structlog(settings=first_options)
        initial = u.create_module_logger("tests.logging.factory.initial")
        first_stdout = io.StringIO()
        with redirect_stderr(first_stdout):
            _ = initial.info("initial output")
        u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
        current_stdout = io.StringIO()
        with redirect_stderr(current_stdout):
            current = u.create_module_logger("tests.logging.factory.synchronous")
            result = current.info("synchronous replacement")
        tm.ok(result.success)
        tm.ok("synchronous replacement" in current_stdout.getvalue())
        tm.fail("synchronous replacement" in old_stream.getvalue())
        tm.fail("synchronous replacement" in first_stdout.getvalue())

    def test_registered_native_wrapper_emits_bound_context(self) -> None:
        """A registered native wrapper executes real context-bound emission."""
        stream = io.StringIO()
        registrations: list[str] = []

        def wrapper_factory() -> type[p.NativeLoggingWrapper]:
            registrations.append("native")
            return structlog.stdlib.BoundLogger

        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                wrapper_class_factory=wrapper_factory,
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        logger = u.create_module_logger("tests.logging.native.wrapper")
        result = logger.bind(operation="native").warning("emitted native event")
        tm.ok(result.success)
        tm.that(registrations, eq=["native"])
        tm.ok("emitted native event" in stream.getvalue())
        tm.ok("operation" in stream.getvalue())
        tm.ok("native" in stream.getvalue())

    def test_native_destinations_share_one_sanitized_observer_event(self) -> None:
        """Native handlers present the same event without duplicating hooks."""
        structured, console = io.StringIO(), io.StringIO()
        password = u.generate_id()
        observed: list[p.LoggingRecord] = []
        handlers = (
            u.create_logging_handler(stream=structured),
            u.create_logging_handler(stream=console, console_renderer=True),
        )
        selected = u.resolve_logging_settings(log_level=c.LogLevel.DEBUG)
        u.configure_structlog(
            runtime_settings=selected,
            settings=m.StructlogOptions(
                handlers=handlers, observers=(observed.append,)
            ),
        )
        with structlog.contextvars.bound_contextvars(correlation_id="real-operation"):
            tm.ok(
                u.log(
                    "delivered %s", "once", payload={"password": password, "count": 2}
                )
            )
        tm.that(len(observed), eq=1)
        record = observed[0]
        rendered = u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(structured.getvalue())))
        tm.that(record.message, eq="delivered once")
        tm.that(rendered["correlation_id"], eq=record.context["correlation_id"])
        sanitized = u.Cli.json_as_mapping(record.context["payload"])
        tm.that(rendered["payload"], eq=sanitized)
        tm.that(sanitized["count"], eq=2)
        tm.fail(sanitized["password"] == password)
        tm.that(str(record.context), lacks=password)
        tm.that(console.getvalue(), has="delivered once", lacks=password)
        tm.that(structured.getvalue(), lacks=password)
        for handler in handlers:
            handler.close()

    def test_ordinary_event_is_filtered_without_relabeling_or_changing_caller(
        self,
    ) -> None:
        """WARNING policy filters ordinary INFO before observers or destinations."""
        first, second = io.StringIO(), io.StringIO()
        observed: list[p.LoggingRecord] = []
        handlers = (
            u.create_logging_handler(stream=first),
            u.create_logging_handler(stream=second),
        )
        selected = u.resolve_logging_settings(log_level=c.LogLevel.WARNING)
        u.configure_structlog(
            runtime_settings=selected,
            settings=m.StructlogOptions(
                handlers=handlers,
                observers=(observed.append,),
            ),
        )
        tm.ok(u.log("ordinary filtered event"))
        tm.that(first.getvalue(), eq="")
        tm.that(second.getvalue(), eq="")
        tm.that(observed, eq=[])
        tm.ok(u.warning("visible warning"))
        tm.that(len(observed), eq=1)
        record = observed[0]
        tm.that(record.level, eq="warning")
        tm.that(record.context[c.ContextKey.LOGGER_NAME], eq=__name__)
        tm.that(record.context["pathname"], eq=__file__)
        tm.that(
            record.context["func_name"],
            eq="test_ordinary_event_is_filtered_without_relabeling_or_changing_caller",
        )
        for stream in (first, second):
            event = u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(stream.getvalue())))
            tm.that(event["event"], eq=record.message)
            tm.that(
                event[c.ContextKey.LOGGER_NAME],
                eq=record.context[c.ContextKey.LOGGER_NAME],
            )

    def test_native_context_ownership_is_thread_and_task_local(self) -> None:
        """Concurrent split scope APIs neither leak keys nor erase a parent."""
        observed: list[p.LoggingRecord] = []
        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                observers=(observed.append,),
                logger_factory=structlog.PrintLoggerFactory(file=io.StringIO()),
            )
        )
        thread_barrier = threading.Barrier(2)

        def emit_thread(marker: str) -> None:
            tm.ok(u.bind_context("isolation", logging_scope_isolation=marker))
            thread_barrier.wait()
            tm.ok(u.log("thread context", marker=marker))
            tm.ok(u.clear_scope("isolation"))
            tm.that(
                structlog.contextvars.get_contextvars().get("logging_scope_isolation"),
                eq=None,
            )

        async def emit_tasks() -> None:
            barrier = asyncio.Barrier(2)

            async def emit(marker: str) -> None:
                tm.ok(u.bind_context("isolation", logging_scope_isolation=marker))
                await barrier.wait()
                tm.ok(u.log("task context", marker=marker))
                tm.ok(u.clear_scope("isolation"))
                tm.that(
                    structlog.contextvars.get_contextvars().get(
                        "logging_scope_isolation"
                    ),
                    eq=None,
                )

            await asyncio.gather(emit("task one"), emit("task two"))

        with structlog.contextvars.bound_contextvars(logging_scope_isolation="parent"):
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(emit_thread, ("thread one", "thread two")))
            asyncio.run(emit_tasks())
            tm.that(
                structlog.contextvars.get_contextvars()["logging_scope_isolation"],
                eq="parent",
            )
        tm.that(len(observed), eq=4)
        for record in observed:
            tm.that(
                record.context["logging_scope_isolation"], eq=record.context["marker"]
            )

    def test_bound_cached_logger_observes_factory_and_hook_removal(self) -> None:
        """Binding keeps context, not a stale processor/factory composition."""
        old, current = io.StringIO(), io.StringIO()
        observed: list[p.LoggingRecord] = []
        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                cache_logger_on_first_use=True,
                logger_factory=structlog.PrintLoggerFactory(file=old),
                observers=(observed.append,),
            )
        )
        cached = u.fetch_logger("real.cached").bind(operation="bound")
        tm.ok(cached.info("before replacement"))
        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                console_renderer=False,
                logger_factory=structlog.PrintLoggerFactory(file=current),
            )
        )
        tm.ok(cached.info("after replacement"))
        tm.that(len(observed), eq=1)
        tm.that(old.getvalue(), lacks="after replacement")
        rendered = u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(current.getvalue())))
        tm.that(rendered["operation"], eq="bound")
        tm.that(rendered["event"], eq="after replacement")

    def test_error_trace_retains_native_chains_without_locals_or_secrets(self) -> None:
        """ERROR plus trace expands the original cause using native processors."""
        stream = io.StringIO()
        observed: list[p.LoggingRecord] = []
        sensitive_value = "selected-sensitive-input"
        selected = u.resolve_logging_settings(
            log_level=c.LogLevel.ERROR, trace=True, debug=False
        )
        u.configure_structlog(
            runtime_settings=selected,
            settings=m.StructlogOptions(
                console_renderer=False,
                observers=(observed.append,),
                redacted_values=(sensitive_value,),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        cause = ValueError(f"invalid credential {sensitive_value}")
        original = RuntimeError("operation failed")
        try:
            try:
                raise cause
            except ValueError:
                raise original from cause
        except RuntimeError as error:
            tm.ok(u.failure("causal error", exception=error))
        tm.ok(u.debug("filtered detail"))
        record = observed[0]
        tm.ok(record.exception is original)
        tm.that(
            "exception"
            in m.LoggingRecord.model_validate(
                record, from_attributes=True
            ).model_dump(),
            eq=False,
        )
        tm.that(record.context["exception_type"], eq=type(original).__name__)
        tm.that(record.context["exception_message"], eq=str(original))
        tm.that(stream.getvalue(), has="stack_trace", lacks=sensitive_value)
        tm.that(stream.getvalue(), has="ValueError", lacks='"locals"')
        tm.that(stream.getvalue(), lacks="filtered detail")
        tm.ok(original.__cause__ is cause)

    def test_business_exception_survives_selected_observer_failure(self) -> None:
        """The selected failure is visible without replacing the original cause."""
        hook_error = OSError("capture unavailable")
        cause = ValueError("business cause")
        original = RuntimeError("business failure")

        def capture(event: p.LoggingRecord) -> Never:
            _ = event
            raise hook_error

        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                observers=(capture,),
                async_logging=False,
            )
        )

        def log_business_failure() -> None:
            try:
                try:
                    raise cause
                except ValueError:
                    raise original from cause
            except RuntimeError as error:
                _ = u.failure("operation failed", exception=error)

        with pytest.raises(RuntimeError) as caught:
            log_business_failure()
        tm.ok(caught.value is original)
        tm.ok(original.__cause__ is cause)
        tm.ok(original.__context__ is hook_error)
        tm.that(original.__notes__, has="Logging also failed: OSError")

    def test_native_exception_group_and_original_cause_remain_observable(self) -> None:
        """Group children survive native expansion without serializing raw causes."""
        observed: list[p.LoggingRecord] = []
        stream = io.StringIO()
        selected = u.resolve_logging_settings(log_level=c.LogLevel.ERROR, trace=True)
        u.configure_structlog(
            runtime_settings=selected,
            settings=m.StructlogOptions(
                console_renderer=False,
                observers=(observed.append,),
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        original = ExceptionGroup(
            "grouped operation", [ValueError("first"), OSError("second")]
        )
        try:
            raise original
        except ExceptionGroup as error:
            tm.ok(u.failure("group failed", exception=error))
        tm.ok(observed[0].exception is original)
        tm.that(stream.getvalue(), has="ExceptionGroup")
        tm.that(stream.getvalue(), has="ValueError")
        tm.that(stream.getvalue(), has="OSError", lacks='"locals"')

    def test_native_handler_sink_failure_escapes(self) -> None:
        """The native StreamHandler adapter does not swallow a closed stream."""
        stream = io.StringIO()
        handler = u.create_logging_handler(stream=stream)
        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                handlers=(handler,),
                async_logging=False,
            )
        )
        stream.close()
        with pytest.raises(ValueError, match="closed file") as caught:
            _ = u.log("native sink failure")
        tm.ok(caught.value.__traceback__ is not None)

    def test_saved_native_exception_info_preserves_original_and_supplied_traceback(
        self,
    ) -> None:
        """Outside except, a failed sink never replaces a saved exception tuple."""

        def fail_with_cause(
            original: ValueError | KeyboardInterrupt, cause: LookupError
        ) -> Never:
            raise original from cause

        def fail_again(original: ValueError | KeyboardInterrupt) -> Never:
            raise original

        for original in (
            ValueError("saved business failure"),
            KeyboardInterrupt("saved interrupt"),
        ):
            cause = LookupError("original cause")
            try:
                fail_with_cause(original, cause)
            except (ValueError, KeyboardInterrupt):
                saved_tb = original.__traceback__
                saved: t.LoggingExceptionInfo = type(original), original, saved_tb
                resolved = u.resolve_logging_exception_info(True)
                if resolved is None:
                    pytest.fail("An active exception must resolve to native exc_info")
                tm.ok(resolved[0] is saved[0])
                tm.ok(resolved[1] is saved[1])
                tm.ok(resolved[2] is saved[2])
                tm.that(u.resolve_logging_exception_info(False), eq=None)
            else:
                pytest.fail("The original exception must be raised with its cause")
            try:
                fail_again(original)
            except (ValueError, KeyboardInterrupt):
                discarded_tb = original.__traceback__
            else:
                pytest.fail("Re-raising the original exception must add a traceback")
            tm.fail(discarded_tb is saved_tb)
            stream = io.StringIO()
            observed: list[p.LoggingRecord] = []
            handler = u.create_logging_handler(stream=stream)
            u.configure_structlog(
                settings=m.StructlogOptions(
                    log_level=logging.DEBUG,
                    handlers=(handler,),
                    observers=(observed.append,),
                )
            )
            stream.close()
            with pytest.raises(type(original)) as caught:
                _ = u.fetch_logger("saved.native.exception").error(
                    "failed operation", exc_info=saved
                )
            tm.ok(caught.value is original)
            tm.ok(observed[0].exception is original)
            tm.ok(original.__cause__ is cause)
            tm.ok(isinstance(original.__context__, ValueError))
            restored_tb = original.__traceback__
            while restored_tb is not None and restored_tb is not saved_tb:
                tm.fail(restored_tb is discarded_tb)
                restored_tb = restored_tb.tb_next
            tm.ok(restored_tb is saved_tb)

    def test_declared_sensitive_keys_are_casefolded_before_hooks_and_sinks(
        self,
    ) -> None:
        """One shared redactor handles actual keys and embedded JSON uniformly."""
        structured, console = io.StringIO(), io.StringIO()
        observed: list[p.LoggingRecord] = []
        declared_key = "ClientSecret"
        sensitive_value = "mixed-case-sensitive-payload"
        options = m.StructlogOptions(
            log_level=logging.DEBUG,
            redacted_keys=(declared_key,),
            observers=(observed.append,),
            handlers=(
                u.create_logging_handler(stream=structured),
                u.create_logging_handler(stream=console, console_renderer=True),
            ),
        )
        u.configure_structlog(settings=options)
        payload: t.JsonDict = dict.fromkeys(
            (declared_key, declared_key.casefold(), declared_key.swapcase()),
            sensitive_value,
        )
        nested: list[t.JsonPayload] = [payload]
        tm.ok(
            u.log(
                "declared sensitive payload",
                payload={
                    "nested": nested,
                    "encoded": tm.ok(u.Cli.json_dumps(payload)),
                    "retained": "public",
                },
            )
        )
        tm.that(len(observed), eq=1)
        dumped = m.LoggingRecord.model_validate(
            observed[0], from_attributes=True
        ).model_dump_json()
        for text in (structured.getvalue(), console.getvalue(), dumped):
            tm.that(text, lacks=sensitive_value, has="public")

    def test_invalid_policy_and_unbound_selected_handler_have_no_emission(self) -> None:
        """Validation rejects selected incomplete composition before effects."""
        original = u.logging_settings()
        stream = io.StringIO()
        with pytest.raises(m.ValidationError):
            _ = u.resolve_logging_settings(logging_profile="unbound-presentation")
        with pytest.raises(TypeError, match="ProcessorFormatter"):
            u.configure_structlog(
                settings=m.StructlogOptions(handlers=(logging.StreamHandler(stream),))
            )
        tm.ok(u.logging_settings() is original)
        tm.that(stream.getvalue(), eq="")

    def test_native_positional_formatter_accepts_mapping_and_filters_before_format(
        self,
    ) -> None:
        """Native formatting follows severity election, not a second custom path."""
        stream = io.StringIO()
        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.INFO,
                console_renderer=False,
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            )
        )
        logger = u.fetch_logger("native.arguments")
        tm.ok(logger.debug("filtered invalid %d", "not an integer"))
        tm.that(stream.getvalue(), eq="")
        tm.ok(logger.info("named %(actor)s", {"actor": "sam"}))
        event = u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(stream.getvalue())))
        tm.that(event["event"], eq="named sam")
        tm.fail("positional_args" in event)

    @staticmethod
    def exercise_async_stream_failure() -> None:
        """Observe a real closed stream in an isolated logging process."""
        stream = io.StringIO()
        stream.close()
        u.configure_structlog(settings=m.StructlogOptions(async_logging=True))
        logger = u.create_module_logger("tests.logging.async.failure")
        with redirect_stderr(stream):
            with pytest.raises(ValueError, match="closed file") as first:
                _ = logger.info("first closed output")
            with pytest.raises(ValueError, match="closed file") as second:
                _ = logger.info("second closed output")
        tm.ok(second.value is first.value)
        tm.ok(first.value.__traceback__ is not None)
        with pytest.raises(ValueError, match="closed file") as teardown:
            u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
        tm.ok(teardown.value is first.value)

    @staticmethod
    def assert_blocked_pipe_outcome(outcome: p.LogResult | ValueError) -> None:
        """Validate the public outcome after a blocked writer recovers."""
        if isinstance(outcome, ValueError):
            tm.that(str(outcome), eq="I/O operation on stopped async log writer")
            tm.ok(outcome.__traceback__ is not None)
        else:
            tm.ok(outcome.success)

    @staticmethod
    def exercise_blocked_pipe_shutdown() -> None:
        """A public synchronous transition bounds a blocked native pipe.

        Raises:
            TimeoutError: If public emission cannot start or recover in time.

        """
        read_fd, write_fd = os.pipe()
        capacity = fcntl.fcntl(write_fd, fcntl.F_GETPIPE_SZ)
        payload = "x" * (capacity * 3)
        try:
            with os.fdopen(write_fd, "w") as stream, redirect_stderr(stream):
                u.configure_structlog(
                    settings=m.StructlogOptions(
                        async_logging=True, console_renderer=False
                    )
                )
                logger = u.create_module_logger("tests.logging.async.blocked")
                emitted: list[p.LogResult | ValueError] = []

                def emit() -> None:
                    try:
                        result = logger.info("blocked pipe output %s", payload)
                    except ValueError as exc:
                        emitted.append(exc)
                    else:
                        emitted.append(result)

                emission = threading.Thread(target=emit, daemon=True)
                emission.start()
                write_fds: list[int] = []
                exceptional_fds: list[int] = []
                readable, _, _ = select.select(
                    [read_fd], write_fds, exceptional_fds, c.DEFAULT_TIMEOUT_SECONDS
                )
                if not readable:
                    msg = "Public logging did not reach the native pipe"
                    raise TimeoutError(msg)
                chunks = [os.read(read_fd, capacity)]
                tm.ok(b"blocked pipe output" in chunks[0])
                with pytest.raises(TimeoutError, match="shutdown timeout") as stopped:
                    u.configure_structlog(
                        settings=m.StructlogOptions(async_logging=False)
                    )
                tm.ok(stopped.value.__traceback__ is not None)

                def drain() -> None:
                    while chunk := os.read(read_fd, capacity):
                        chunks.append(chunk)

                reader = threading.Thread(target=drain, daemon=True)
                reader.start()
                emission.join(timeout=c.DEFAULT_TIMEOUT_SECONDS)
                if emission.is_alive():
                    msg = "Public logging did not recover after the pipe drained"
                    raise TimeoutError(msg)
                tm.that(len(emitted), eq=1)
                TestsFlextLoggingStages.assert_blocked_pipe_outcome(emitted[0])
                u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
            reader.join(timeout=c.DEFAULT_TIMEOUT_SECONDS)
            if reader.is_alive():
                msg = "Native pipe drainage did not finish after closing its writer"
                raise TimeoutError(msg)
            tm.ok(payload.encode() in b"".join(chunks))
        finally:
            os.close(read_fd)

    @pytest.mark.parametrize("scenario", ["stream_failure", "blocked_pipe"])
    def test_async_public_failure_and_shutdown(self, scenario: str) -> None:
        """Original failures and bounded shutdown cross public boundaries.

        Raises:
            TimeoutError: If the child exceeds the configured process deadline.

        """
        match scenario:
            case "blocked_pipe":
                exercise = self.exercise_blocked_pipe_shutdown
            case _:
                exercise = self.exercise_async_stream_failure
        process = multiprocessing.get_context("spawn").Process(
            target=exercise,
        )
        process.start()
        try:
            process.join(timeout=c.DEFAULT_TIMEOUT_SECONDS)
            if process.is_alive():
                process.terminate()
                process.join(timeout=c.LOG_SHUTDOWN_JOIN_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=c.LOG_SHUTDOWN_JOIN_SECONDS)
                msg = "Public async logging process exceeded its configured timeout"
                raise TimeoutError(msg)
            tm.that(process.exitcode, eq=0)
        finally:
            if not process.is_alive():
                process.close()
