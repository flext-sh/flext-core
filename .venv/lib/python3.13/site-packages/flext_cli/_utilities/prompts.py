"""Prompt helpers shared through ``u.Cli``."""

from __future__ import annotations

from flext_cli import c, p, r, t


class FlextCliUtilitiesPrompts:
    """Pure prompt formatting and validation helpers."""

    @staticmethod
    def prompts_confirmation_text(message: str, *, default: bool) -> str:
        """Build one standardized confirmation prompt message."""
        return (
            f"{message}{c.Cli.PROMPT_CONFIRM_YES}"
            if default
            else f"{message}{c.Cli.PROMPT_CONFIRM_NO}"
        )

    @staticmethod
    def prompts_display_message(message: str, default: str) -> str:
        """Build prompt display message with optional default marker."""
        if default:
            return f"{message}{c.Cli.PROMPT_DEFAULT_FMT.format(default=default)}"
        return message

    @staticmethod
    def prompts_effective_text(raw_input: str, default: str) -> str:
        """Normalize one raw input text to one effective prompt value."""
        trimmed = raw_input.strip()
        return trimmed or default

    @staticmethod
    def prompts_parse_confirmation(text: str, *, default: bool) -> bool | None:
        """Parse one confirmation input text into yes/no/default/invalid."""
        normalized = text.strip().lower()
        if not normalized:
            return default
        if normalized in c.Cli.PROMPT_YES_VALUES:
            return True
        if normalized in c.Cli.PROMPT_NO_VALUES:
            return False
        return None

    @staticmethod
    def prompts_choice_result(
        *, interactive: bool, choices: t.StrSequence, default: str | None
    ) -> p.Result[str]:
        """Validate one choice prompt contract and return one canonical result."""
        result: p.Result[str]
        if not choices:
            result = r[str].fail(c.Cli.ERR_NO_CHOICES)
        elif not interactive:
            if default and default in choices:
                result = r[str].ok(default)
            else:
                result = r[str].fail(c.Cli.ERR_INTERACTIVE_CHOICE_DISABLED)
        elif default is None:
            result = r[str].fail(
                c.Cli.ERR_CHOICE_REQUIRED_FMT.format(choices=", ".join(choices))
            )
        elif default not in choices:
            result = r[str].fail(c.Cli.ERR_INVALID_CHOICE_FMT.format(choice=default))
        else:
            result = r[str].ok(default)
        return result

    @staticmethod
    def prompts_password_result(password: str, *, min_length: int) -> p.Result[str]:
        """Validate one password length contract."""
        if len(password) < min_length:
            return r[str].fail(
                c.Cli.ERR_PASSWORD_TOO_SHORT_FMT.format(min_length=min_length)
            )
        return r[str].ok(password)


__all__: t.MutableSequenceOf[str] = ["FlextCliUtilitiesPrompts"]
