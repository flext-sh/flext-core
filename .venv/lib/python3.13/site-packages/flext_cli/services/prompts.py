"""User interaction tools for CLI applications."""

from __future__ import annotations

from typing import override

from flext_cli import c, m, p, r, t, u
from flext_cli.services._prompts_support import FlextCliPromptsSupport

# NOTE (multi-agent): mro-i6nq.13 — consolidated _prompts_parts/part_01+part_02
# (and the empty part_03 + facade pass-through layers) into this single cohesive
# module. Repeated try/except CLI_SAFE_EXCEPTIONS -> _fatal -> r.fail idioms are
# collapsed onto the support base ``_guarded`` (u.guard_result). Public surface
# and error messages unchanged.


class FlextCliPrompts(FlextCliPromptsSupport):
    """Interactive CLI prompt surface exposed through the CLI service runtime."""

    @override
    def execute(self) -> p.Result[m.Cli.RuntimeStatus]:
        """Return the current CLI runtime status."""
        return r[m.Cli.RuntimeStatus].ok(u.Cli.cmd_status())

    def confirm(self, message: str, *, default: bool = False) -> p.Result[bool]:
        """Read a yes/no confirmation or return the configured default."""
        try:
            if self.state.quiet or not self.state.interactive:
                return r[bool].ok(default)
            prompt_text = u.Cli.prompts_confirmation_text(message, default=default)
            return self._read_confirmation_input(message, prompt_text, default=default)
        except KeyboardInterrupt:
            return r[bool].fail(c.Cli.ERR_USER_CANCELLED_CONFIRMATION)
        except EOFError:
            return r[bool].fail(c.Cli.ERR_INPUT_STREAM_ENDED)
        except c.Cli.CLI_SAFE_EXCEPTIONS as exc:
            self._fatal("confirm", message, exc, "Confirmation failed completely")
            return r[bool].fail(c.Cli.ERR_CONFIRMATION_FAILED_FMT.format(error=exc))

    def prompt(self, message: str, default: str = "") -> p.Result[str]:
        """Read one text value or return the configured default."""
        if self.state.quiet or not self.state.interactive:
            return r[str].ok(default)
        return self._guarded(
            "prompt",
            message,
            lambda: r[str].ok(self._read_prompt_value(message, default)),
            consequence="Prompt failed completely",
            error_format=c.Cli.ERR_PROMPT_FAILED_FMT,
        )

    def prompt_choice(
        self, message: str, choices: t.StrSequence, default: str | None = None
    ) -> p.Result[str]:
        """Read one value constrained to the supplied choices."""
        return self._guarded(
            "prompt_choice",
            message,
            lambda: u.Cli.prompts_choice_result(
                interactive=self.state.interactive, choices=choices, default=default
            ),
            consequence="Choice prompt failed completely",
            error_format=c.Cli.ERR_CHOICE_PROMPT_FAILED_FMT,
        )

    def prompt_password(
        self,
        message: str = "Password:",
        min_length: int = c.Cli.PROMPT_MIN_PASSWORD_LENGTH,
    ) -> p.Result[str]:
        """Read a password and enforce the minimum length."""
        if not self.state.interactive:
            return r[str].fail(c.Cli.ERR_INTERACTIVE_PASSWORD_DISABLED)
        return self._guarded(
            "prompt_password",
            message,
            lambda: u.Cli.prompts_password_result(
                self._password_reader(f"{message}{c.Cli.PROMPT_SPACE}"),
                min_length=min_length,
            ),
            consequence="Password prompt failed completely",
            error_format=c.Cli.ERR_PASSWORD_PROMPT_FAILED_FMT,
        )

    def print_error(self, message: str) -> p.Result[bool]:
        """Render an error message through the canonical prompt output path."""
        return self._print_message(
            message,
            c.LogLevel.ERROR,
            c.Cli.PROMPT_ERROR_FMT,
            "Print error failed: {error}",
        )

    def print_success(self, message: str) -> p.Result[bool]:
        """Render a success message through the canonical prompt output path."""
        return self._print_message(
            message,
            c.LogLevel.INFO,
            c.Cli.PROMPT_SUCCESS_FMT,
            "Print success failed: {error}",
        )

    def print_warning(self, message: str) -> p.Result[bool]:
        """Render a warning message through the canonical prompt output path."""
        return self._print_message(
            message,
            c.LogLevel.WARNING,
            c.Cli.PROMPT_WARNING_FMT,
            "Print warning failed: {error}",
        )

    def _read_prompt_value(self, message: str, default: str) -> str:
        """Read one prompt value and record the canonical prompt log."""
        display_message = u.Cli.prompts_display_message(message, default)
        raw = self._input_reader(f"{display_message}{c.Cli.PROMPT_SEP}")
        value: str = u.Cli.prompts_effective_text(raw, default)
        if not self._is_test_env():
            self._log(
                c.LogLevel.INFO,
                c.Cli.PROMPT_LOG_FMT.format(message=message, input=value),
            )
        return value


__all__: list[str] = ["FlextCliPrompts"]
