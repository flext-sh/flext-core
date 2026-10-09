"""Typed stage registration through the real public logging boundary.

Copyright (c) 2026 FLEXT Team. All rights reserved.
tests/unit/test_logging_stages
SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import io
import logging
import traceback
from typing import TYPE_CHECKING

import pytest
import structlog
from flext_tests import tm
from pydantic import ValidationError

from flext_core import m, p, t, u

if TYPE_CHECKING:
    from collections.abc import Generator


class TestsFlextLoggingStages:
    """Observable contracts of typed event stages and terminal rendering."""

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

        result = logger.info("ordered stages", payload=payload)

        tm.that(result.success, eq=True)
        tm.that(len(observed), eq=2)
        tm.that(observed[1]["payload"], eq=payload)
        tm.that(observed[1]["exc_info"], eq=True)
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
