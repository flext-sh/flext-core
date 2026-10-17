"""Typed stage registration through the real public logging boundary.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/unit/test_logging_stages
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import fcntl
import io
import logging
import multiprocessing
import os
import select
import sys
import threading
import traceback
from contextlib import redirect_stdout
from typing import TYPE_CHECKING

import pytest
import structlog
from pydantic import ValidationError

from tests import c, m, p, t, tm, u

if TYPE_CHECKING:
    from collections.abc import Generator


class TestsFlextLoggingStages:
    """Observable contracts of typed event stages and terminal rendering."""

    @staticmethod
    def test_closed_async_stream_preserves_native_failure() -> None:
        """A failed sink terminates without logging back into its own writer."""
        consumer = (
            "import logging, sys, structlog\n"
            "from flext_core import m, u\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=True, log_level=logging.INFO))\n"
            "sys.stdout.close()\n"
            "structlog.get_logger().info('closed stream')\n"
        )
        executed = tm.ok(u.Cli.run_raw((sys.executable, "-c", consumer)))
        tm.that(u.Cli.process_succeeded(executed.outcome), eq=False)
        tm.that(executed.outcome.timed_out, eq=False)
        tm.that(executed.outcome.forwarded_signal, eq=None)
        tm.that(executed.stderr, has="ValueError: I/O operation on closed file.")

    @staticmethod
    def test_async_emission_drains_before_reconfiguration_and_process_exit() -> None:
        """Accepted public events reach the native stream before shutdown."""
        events = (
            "first accepted event",
            "second accepted event",
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
        tm.that(u.Cli.process_succeeded(executed.outcome), eq=True)
        tm.that(executed.outcome.timed_out, eq=False)
        emitted = tuple(
            u.Cli.json_as_mapping(tm.ok(u.Cli.json_parse(line)))["event"]
            for line in executed.stdout.splitlines()
        )
        tm.that(emitted, eq=events)
        tm.that(executed.stderr, empty=True)

    @staticmethod
    def test_failed_async_sink_rethrows_the_same_public_exception() -> None:
        """Subsequent public writes preserve the first native stream exception."""
        consumer = (
            "import logging, sys\n"
            "from flext_core import m, u\n"
            "u.configure_structlog(settings=m.StructlogOptions("
            "async_logging=True, log_level=logging.INFO))\n"
            "logger = u.create_module_logger('tests.logging.native.failure')\n"
            "sys.stdout.close()\n"
            "try:\n"
            "    logger.info('first failed event')\n"
            "except ValueError as first:\n"
            "    try:\n"
            "        logger.info('second failed event')\n"
            "    except ValueError as second:\n"
            "        if second is not first:\n"
            "            raise AssertionError('native failure identity changed')\n"
            "        raise first\n"
            "    raise AssertionError('second event did not fail')\n"
            "raise AssertionError('closed native stream did not fail')\n"
        )
        executed = tm.ok(u.Cli.run_raw((sys.executable, "-c", consumer)))
        tm.that(u.Cli.process_succeeded(executed.outcome), eq=False)
        tm.that(executed.outcome.timed_out, eq=False)
        tm.that(executed.outcome.forwarded_signal, eq=None)
        tm.that(executed.stderr, has="ValueError: I/O operation on closed file.")
        tm.that(executed.stderr, lacks="AssertionError")

    @pytest.fixture(autouse=True)
    def restore_logging_options(self) -> Generator[None]:
        """Restore public default configuration after each registration."""
        yield
        u.configure_structlog(settings=m.StructlogOptions())

    def test_ordered_stages_preserve_nested_values_and_render(self) -> None:
        """Stages share an event in order and the final renderer emits it."""
        stream = io.StringIO()
        observed: list[t.LoggingEvent] = []

        def first(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
            _ = logger, method_name
            event_dict["stage_order"] = ["first"]
            observed.append(dict(event_dict))
            return event_dict

        def second(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
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

        try:
            _ = int("active exception context")
        except ValueError:
            result = logger.info("ordered stages", payload=payload, exc_info=True)

        tm.that(result.success, eq=True)
        tm.that(len(observed), eq=2)
        tm.that(observed[1]["payload"], eq=payload)
        # An info event carries no exception context; only the exception
        # entrypoint attaches it (observed at runtime: the stage sees
        # event, level, payload, source and timestamp).
        tm.that("exc_info" in observed[1], eq=False)
        tm.that('"stage_order": ["first", "second"]' in stream.getvalue(), eq=True)

    @pytest.mark.parametrize("entrypoint", ["info", "exception", "trace"])
    def test_first_stage_failure_escapes_unchanged(self, entrypoint: str) -> None:
        """Failure stops the chain and preserves the original exception."""
        stream = io.StringIO()
        visited: list[str] = []
        failure = ValueError("typed stage failed")

        def fail_first(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
            _ = logger, method_name, event_dict
            visited.append("first")
            raise failure

        def later(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
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

        match entrypoint:
            case "exception":
                emit = logger.exception
            case "trace":
                emit = logger.trace
            case _:
                emit = logger.info
        with pytest.raises(ValueError, match="typed stage failed") as caught:
            _ = emit("stage failure")

        tm.that(caught.value is failure, eq=True)
        tm.that(visited, eq=["first"])
        tm.that(stream.getvalue(), eq="")
        frames = traceback.extract_tb(caught.value.__traceback__)
        tm.that(frames[-1].name, eq="fail_first")

    def test_exception_context_reaches_stage(self) -> None:
        """Explicit public exceptions retain their structured traceback."""
        observed: list[t.LoggingEvent] = []
        stream = io.StringIO()

        def observe(
            logger: p.OutputLogger,
            method_name: str,
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
            _ = logger, method_name
            observed.append(dict(event_dict))
            return event_dict

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
            result = logger.exception(
                "operation failed",
                exception=exc,
                operation="read",
            )

        tm.that(result.success, eq=True)
        tm.that(observed[0]["exception_type"], eq=type(error).__name__)
        tm.that(observed[0]["exception_message"], eq=str(error))
        tm.that(str(error) in str(observed[0]["stack_trace"]), eq=True)
        tm.that("operation failed" in stream.getvalue(), eq=True)

    def test_invalid_registration_is_rejected(self) -> None:
        """The options boundary rejects a noncallable stage."""
        with pytest.raises(ValidationError):
            _ = m.StructlogOptions.model_validate({"processing_stages": [1]})

    def test_invalid_callable_signature_raises_at_emission(self) -> None:
        """An untyped registration cannot hide its native invocation error."""
        stream = io.StringIO()

        def invalid_stage() -> t.LoggingEvent:
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

    def test_explicit_registration_preserves_cached_logger_chain(self) -> None:
        """Only fresh loggers consume a replacement processing chain."""
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
            event_dict: t.LoggingEvent,
        ) -> t.LoggingEvent:
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

        tm.that(result.success, eq=True)
        tm.that(observed, eq=["replacement chain"])
        tm.that("cached chain" in stream.getvalue(), eq=True)
        tm.that("replacement chain" in stream.getvalue(), eq=True)

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
        with redirect_stdout(first_stdout):
            _ = initial.info("initial output")
        u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
        current_stdout = io.StringIO()
        with redirect_stdout(current_stdout):
            current = u.create_module_logger("tests.logging.factory.synchronous")
            result = current.info("synchronous replacement")
        tm.that(result.success, eq=True)
        tm.that("synchronous replacement" in current_stdout.getvalue(), eq=True)
        tm.that("synchronous replacement" in old_stream.getvalue(), eq=False)
        tm.that("synchronous replacement" in first_stdout.getvalue(), eq=False)

    def test_registered_native_wrapper_emits_bound_context(self) -> None:
        """A registered native wrapper executes real context-bound emission."""
        stream = io.StringIO()
        registrations: list[str] = []

        def wrapper_factory() -> type[p.NativeLoggingWrapper]:
            registrations.append("native")
            return structlog.BoundLogger

        u.configure_structlog(
            settings=m.StructlogOptions(
                log_level=logging.DEBUG,
                wrapper_class_factory=wrapper_factory,
                logger_factory=structlog.PrintLoggerFactory(file=stream),
            ),
        )
        logger = u.create_module_logger("tests.logging.native.wrapper")
        result = logger.bind(operation="native").warning("emitted native event")
        tm.that(result.success, eq=True)
        tm.that(registrations, eq=["native"])
        tm.that("emitted native event" in stream.getvalue(), eq=True)
        tm.that("operation" in stream.getvalue(), eq=True)
        tm.that("native" in stream.getvalue(), eq=True)

    @staticmethod
    def exercise_async_stream_failure() -> None:
        """Observe a real closed stream in an isolated logging process."""
        stream = io.StringIO()
        stream.close()
        u.configure_structlog(settings=m.StructlogOptions(async_logging=True))
        logger = u.create_module_logger("tests.logging.async.failure")
        with redirect_stdout(stream):
            with pytest.raises(ValueError, match="closed file") as first:
                _ = logger.info("first closed output")
            with pytest.raises(ValueError, match="closed file") as second:
                _ = logger.info("second closed output")
        tm.that(second.value is first.value, eq=True)
        tm.that(first.value.__traceback__ is not None, eq=True)
        with pytest.raises(ValueError, match="closed file") as teardown:
            u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
        tm.that(teardown.value is first.value, eq=True)

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
            with os.fdopen(write_fd, "w") as stream, redirect_stdout(stream):
                u.configure_structlog(
                    settings=m.StructlogOptions(
                        async_logging=True, console_renderer=False
                    )
                )
                logger = u.create_module_logger("tests.logging.async.blocked")
                emitted: list[t.LogResult | ValueError] = []

                def emit() -> None:
                    try:
                        result = logger.info("blocked pipe output %s", payload)
                    except ValueError as exc:
                        emitted.append(exc)
                    else:
                        emitted.append(result)

                emission = threading.Thread(target=emit, daemon=True)
                emission.start()
                readable, _, _ = select.select(
                    [read_fd], [], [], c.DEFAULT_TIMEOUT_SECONDS
                )
                if not readable:
                    msg = "Public logging did not reach the native pipe"
                    raise TimeoutError(msg)
                chunks = [os.read(read_fd, capacity)]
                tm.that(b"blocked pipe output" in chunks[0], eq=True)
                with pytest.raises(TimeoutError, match="shutdown timeout") as stopped:
                    u.configure_structlog(
                        settings=m.StructlogOptions(async_logging=False)
                    )
                tm.that(stopped.value.__traceback__ is not None, eq=True)

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
                outcome = emitted[0]
                if isinstance(outcome, ValueError):
                    tm.that(
                        str(outcome),
                        eq="I/O operation on stopped async log writer",
                    )
                    tm.that(outcome.__traceback__ is not None, eq=True)
                else:
                    tm.that(outcome.success, eq=True)
                u.configure_structlog(settings=m.StructlogOptions(async_logging=False))
            reader.join(timeout=c.DEFAULT_TIMEOUT_SECONDS)
            if reader.is_alive():
                msg = "Native pipe drainage did not finish after closing its writer"
                raise TimeoutError(msg)
            tm.that(payload.encode() in b"".join(chunks), eq=True)
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
