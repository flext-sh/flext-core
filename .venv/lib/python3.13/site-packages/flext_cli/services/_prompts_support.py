"""Prompt service support primitives.

NOTE (multi-agent): mro-i6nq.13 — moved here from the removed
``_prompts_parts/flextcliprompts_support.py`` so the whole numbered
``_prompts_parts`` package could be eliminated. Adds the ``_guarded`` DRY
helper that collapses the repeated ``try/except CLI_SAFE_EXCEPTIONS -> _fatal
-> r.fail(fmt)`` idiom into one canonical ``u.guard_result`` boundary.
"""

from __future__ import annotations

import getpass
from typing import TYPE_CHECKING, Annotated, Self

from flext_cli import c, m, p, r, s, settings, t, u

if TYPE_CHECKING:
    from collections.abc import Callable


class _PromptInputReaderDefault:
    """Resolve the built-in input reader without storing a method descriptor."""

    def __call__(self, prompt: str) -> str:
        return input(prompt)


class _PromptPasswordReaderDefault:
    """Resolve the password reader without storing a method descriptor."""

    def __call__(self, prompt: str) -> str:
        return getpass.getpass(prompt)


class FlextCliPromptsSupport(s):
    """Support owner for prompt runtime state, logging, and input readers."""

    state: Annotated[
        m.Cli.PromptRuntimeState,
        m.Field(description="Prompt runtime state for interaction behavior."),
    ] = m.Field(m.Cli.PromptRuntimeState(), validate_default=True)

    _input_reader: t.Cli.PromptTextReader = m.PrivateAttr(_PromptInputReaderDefault())

    _password_reader: t.Cli.PromptTextReader = m.PrivateAttr(
        _PromptPasswordReaderDefault()
    )

    _test_env_override: bool | None = m.PrivateAttr(None)

    def configure(self, state: m.Cli.PromptRuntimeState) -> Self:
        """Replace prompt runtime state using the canonical CLI model."""
        self.state = state
        return self

    def _is_test_env(self) -> bool:
        """Whether prompt logging must use test-safe behavior.

        The override private attr wins when set (tests pin it via
        ``override_test_env``); otherwise delegate to the canonical
        ``u.Cli.cli_test_env`` utility — settings stay pure flat data (§2.6),
        detection logic lives in the utilities layer, never reimplemented here.
        """
        if self._test_env_override is not None:
            return self._test_env_override
        return u.Cli.cli_test_env(settings)

    def _guarded[TResult](
        self,
        operation: str,
        message: str,
        work: Callable[[], p.Result[TResult]],
        *,
        consequence: str,
        error_format: str,
    ) -> p.Result[TResult]:
        """Run a Result-returning prompt operation behind one exception boundary.

        Collapses the canonical ``try/except CLI_SAFE_EXCEPTIONS -> _fatal ->
        r.fail(fmt)`` idiom shared by every interactive prompt method into a
        single ``u.guard_result`` call plus structured fatal logging.
        """
        guarded = u.guard_result(
            work, catch=c.Cli.CLI_SAFE_EXCEPTIONS, op_name=operation
        )
        if guarded.success:
            return guarded
        exc = guarded.error or operation
        self._fatal(operation, message, Exception(exc), consequence)
        return r[TResult].fail(error_format.format(error=exc))

    def _fatal(
        self, operation: str, message: str, exc: Exception, consequence: str
    ) -> None:
        self._log(
            c.LogLevel.ERROR,
            f"FATAL ERROR during {operation} - operation aborted",
            operation=operation,
            prompt_message=message,
            error=str(exc),
            error_type=type(exc).__name__,
            consequence=consequence,
            severity="critical",
        )

    def _log(self, log_level: str, message: str, **context: t.LogValue) -> None:
        match log_level:
            case c.LogLevel.DEBUG:
                self.logger.debug(message, **context)
            case c.LogLevel.ERROR:
                self.logger.error(message, **context)
            case c.LogLevel.WARNING:
                self.logger.warning(message, **context)
            case _:
                self.logger.info(message, **context)

    def _print_message(
        self,
        message: str,
        log_level: str,
        message_format: str,
        error_message_template: str,
    ) -> p.Result[bool]:
        try:
            formatted_message = message_format.format(message=message)
            self._log(log_level, formatted_message)
            return r[bool].ok(True)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            self.logger.exception(
                "FAILED to print message - operation aborted",
                operation="_print_message",
                log_level=log_level,
                prompt_message=message,
                error=str(exc),
                error_type=type(exc).__name__,
                consequence="Message not displayed",
            )
            return r[bool].fail(error_message_template.format(error=exc))

    def _read_confirmation_input(
        self, message: str, prompt_text: str, *, default: bool
    ) -> p.Result[bool]:
        while True:
            input_text = self._input_reader(prompt_text)
            parsed = u.Cli.prompts_parse_confirmation(input_text, default=default)
            if parsed is not None:
                return r[bool].ok(parsed)
            self._log(
                c.LogLevel.WARNING,
                c.Cli.ERR_INVALID_CONFIRM_INPUT,
                operation="confirm",
                prompt_message=message,
                user_input=input_text,
                consequence="Prompting again",
            )


__all__: list[str] = ["FlextCliPromptsSupport"]
