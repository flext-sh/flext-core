"""Manual-command blocker (AGENTS.md `Build & Test`).

Two responsibilities:

- ``command_blocked`` — predicate flagging a bare tool invocation (ruff/pytest/git/…)
  that bypasses the ``make`` / ``python -m flext_infra`` monopoly. Backs both the
  pre-commit hook and the Claude PreToolUse guard. Deny rules are evaluated
  FIRST, per shell segment, after stripping wrappers and path components — an
  allow-list substring can never short-circuit a deny.
- ``render_pre_commit_config`` — the canonical ``.pre-commit-config.yaml`` content
  (hooks call ``uv run --all-packages python -m flext_infra``, never standalone
  scripts).

``execute`` is a drift gate: the live ``.pre-commit-config.yaml`` MUST equal the
rendered canonical template.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, override

from flext_core import r
from flext_infra import c, t, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraManualCommandValidator(s[bool]):
    """Block bare tool invocations in automation and gate pre-commit drift."""

    @classmethod
    def command_blocked(cls, command: str) -> bool:
        """Check whether any shell segment runs a managed tool outside make/flext_infra."""
        stripped = command.strip()
        if not stripped:
            return False
        return any(
            cls._segment_blocked(segment.strip())
            for segment in c.Infra.MANUAL_CMD_SEGMENT_RE.split(stripped)
        )

    @classmethod
    def _segment_blocked(cls, segment: str) -> bool:
        """Apply deny rules to a single shell segment after normalisation."""
        if not segment:
            return False
        try:
            tokens = cls._strip_wrappers(shlex.split(segment))
        except ValueError:
            tokens = cls._strip_wrappers(segment.split())
        if not tokens:
            return False
        head = Path(tokens[0]).name
        rest = tokens[1:]
        blocked_tools = c.Infra.MANUAL_CMD_BLOCKED_TOOLS
        if head in blocked_tools:
            return True
        if (
            head in c.Infra.MANUAL_CMD_RUNNERS
            and cls._module_after_m(rest) in blocked_tools
        ):
            return True
        if head == "git" and rest and rest[0] in c.Infra.MANUAL_CMD_BLOCKED_GIT:
            return True
        if head == "sed" and any(cls._is_sed_inplace(arg) for arg in rest):
            return True
        return head in c.Infra.MANUAL_CMD_REWRITE_TOOLS and any(
            arg in c.Infra.MANUAL_CMD_REWRITE_FLAGS or arg.startswith("--rewrite=")
            for arg in rest
        )

    @classmethod
    def _strip_wrappers(cls, tokens: t.StrSequence) -> t.MutableSequenceOf[str]:
        """Drop leading wrapper commands and ``env VAR=val`` assignments."""
        out = list(tokens)
        while out:
            name = Path(out[0]).name
            if name == "env":
                out = out[1:]
                while out and "=" in out[0] and not out[0].startswith("-"):
                    out = out[1:]
                continue
            if name in c.Infra.MANUAL_CMD_WRAPPERS:
                out = out[1:]
                continue
            if name == "uv" and len(out) > 1 and out[1] == "run":
                out = cls._strip_uv_run_options(out[2:])
                continue
            break
        return out

    @classmethod
    def _strip_uv_run_options(cls, tokens: t.StrSequence) -> t.MutableSequenceOf[str]:
        """Return the real command after ``uv run`` and its options."""
        out = list(tokens)
        while out:
            arg = out[0]
            if arg == "--":
                return out[1:]
            if not arg.startswith("-"):
                return out
            option = arg.split("=", maxsplit=1)[0]
            out = out[1:]
            if (
                "=" not in arg
                and option in c.Infra.MANUAL_CMD_UV_RUN_VALUE_OPTIONS
                and out
            ):
                out = out[1:]
        return out

    @staticmethod
    def _module_after_m(rest: t.StrSequence) -> str:
        """Return the module name following ``-m`` (``python -m <module>``)."""
        for index, arg in enumerate(rest):
            if arg == "-m" and index + 1 < len(rest):
                module_name: str = t.Infra.STR_ADAPTER.validate_python(rest[index + 1])
                return module_name
        return ""

    @staticmethod
    def _is_sed_inplace(arg: str) -> bool:
        """Check whether the argument is a GNU/BSD in-place edit flag (``-i``, ``-i.bak``, ``--in-place``)."""
        return (
            arg == "--in-place"
            or arg.startswith("--in-place=")
            or (arg.startswith("-i") and not arg.startswith("--"))
        )

    @classmethod
    def render_pre_commit_config(cls) -> str:
        """Return the canonical generated ``.pre-commit-config.yaml`` content."""
        config: str = c.Infra.PRE_COMMIT_CONFIG
        return config

    @override
    def execute(self) -> p.Result[bool]:
        """Fail when the live pre-commit config drifts from the canonical template."""
        config_path = self.repository_root / ".pre-commit-config.yaml"
        if not config_path.exists():
            return r[bool].fail(
                ".pre-commit-config.yaml missing — run `make gen` to generate it"
            )
        read = u.Cli.files_read_text(config_path)
        if read.failure:
            return r[bool].fail(read.error or "pre-commit config read failed")
        if read.value.strip() != self.render_pre_commit_config().strip():
            return r[bool].fail(
                ".pre-commit-config.yaml drifted from canonical template — run `make gen`"
            )
        return r[bool].ok(True)


__all__: list[str] = ["FlextInfraManualCommandValidator"]
