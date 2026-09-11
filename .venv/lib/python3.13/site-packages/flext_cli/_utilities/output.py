"""CLI output helpers shared through ``u.Cli``."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import ClassVar

from flext_cli import c, p, t


class FlextCliUtilitiesOutput:
    """Canonical CLI output rendering helpers exposed through ``u.Cli``."""

    # stdout is one process-wide mutable resource. Pipeline stages that carry no
    # dependency between them run concurrently, so two stages can emit a
    # multi-line block at the same time; unguarded `write` interleaves them and
    # the check report becomes unreadable. Serializing only the write keeps the
    # emitted block atomic without constraining the callers.
    _EMIT_LOCK: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def output_resolve_message_type(
        message_type: c.Cli.MessageTypes | None,
    ) -> c.Cli.MessageTypes:
        """Resolve one message type to canonical enum value."""
        return (
            message_type
            if message_type is not None
            else c.Cli.OUTPUT_DEFAULT_MESSAGE_TYPE
        )

    @staticmethod
    def output_resolve_style(style: str | None) -> str:
        """Resolve print style with canonical empty-style fallback."""
        return style if style is not None else c.Cli.OUTPUT_EMPTY_STYLE

    @staticmethod
    def output_message_payload(
        message: str, message_type: c.Cli.MessageTypes | None
    ) -> t.Pair[str, str]:
        """Build one canonical display payload and style from message type."""
        final_type = FlextCliUtilitiesOutput.output_resolve_message_type(message_type)
        default_type = c.Cli.OUTPUT_DEFAULT_MESSAGE_TYPE
        style = c.Cli.MESSAGE_STYLE_MAP.get(
            final_type, c.Cli.MESSAGE_STYLE_MAP[default_type]
        )
        emoji = c.Cli.MESSAGE_EMOJI_MAP.get(
            final_type, c.Cli.MESSAGE_EMOJI_MAP[default_type]
        )
        return f"{emoji} {message}", style

    @staticmethod
    def output_progress_line(
        current: int, total: int, label: str, *, detail: str
    ) -> str:
        """Build one canonical progress line text."""
        width = len(str(total))
        suffix = f" {detail}" if detail else ""
        return f"[{current:0{width}d}/{total}] {label}{suffix}"

    @staticmethod
    def output_summary_content(
        *, total: int, success: int, failed: int, skipped: int
    ) -> str:
        """Build one canonical summary content string."""
        return (
            f"Total: {total}  Success: {success}  Failed: {failed}  Skipped: {skipped}"
        )

    @staticmethod
    def output_debug_line(message: str) -> t.Pair[str, str]:
        """Build one canonical debug line and style."""
        return f"[{c.Cli.OUTPUT_LOG_LEVEL_DEBUG}] {message}", c.Cli.MessageStyles.DIM

    @staticmethod
    def output_table_error(error_message: str | None) -> t.Pair[str, str]:
        """Build one canonical table error line and style."""
        error = error_message or c.Cli.ERR_UNKNOWN_ERROR
        return f"{c.Cli.OUTPUT_TABLE_ERROR_LABEL} {error}", c.Cli.MessageStyles.BOLD_RED

    @staticmethod
    def output_status_line(
        label: str, detail: str, *, success: bool, elapsed: float | None
    ) -> t.Pair[str, str]:
        """Build one canonical status line and style."""
        symbol = c.Cli.SYMBOL_SUCCESS_MARK if success else c.Cli.SYMBOL_FAILURE_MARK
        style = (
            c.Cli.MessageStyles.BOLD_GREEN if success else c.Cli.MessageStyles.BOLD_RED
        )
        timing = f"  ({elapsed:.2f}s)" if elapsed is not None else ""
        line = f"  {symbol} {label:<8} {detail:<24}{timing}"
        return line, style

    @staticmethod
    def output_gate_line(name: str, *, passed: bool, message: str) -> t.Pair[str, str]:
        """Build one canonical gate line and style."""
        symbol = c.Cli.SYMBOL_SUCCESS_MARK if passed else c.Cli.SYMBOL_FAILURE_MARK
        style = (
            c.Cli.MessageStyles.BOLD_GREEN if passed else c.Cli.MessageStyles.BOLD_RED
        )
        suffix = f"  {message}" if message else ""
        return f"    {symbol} {name:<10}{suffix}", style

    @staticmethod
    def emit_raw(text: str) -> None:
        """Write raw text to stdout as one atomic block."""
        with FlextCliUtilitiesOutput._EMIT_LOCK:
            _ = sys.stdout.write(text)
            _ = sys.stdout.flush()

    @classmethod
    def info(cls, msg: str) -> None:
        """Emit one canonical info line."""
        cls.emit_raw(f"{c.Cli.OUTPUT_LOG_LEVEL_INFO}: {msg}\n")

    @classmethod
    def error(cls, msg: str, detail: str | None = None) -> None:
        """Emit one canonical error line with optional detail."""
        cls.emit_raw(f"{c.Cli.OUTPUT_LOG_LEVEL_ERROR}: {msg}\n")
        if detail:
            cls.emit_raw(f"  {detail}\n")

    @classmethod
    def warning(cls, msg: str) -> None:
        """Emit one canonical warning line."""
        cls.emit_raw(f"{c.Cli.OUTPUT_LOG_LEVEL_WARNING}: {msg}\n")

    @classmethod
    def debug(cls, msg: str) -> None:
        """Emit one canonical debug line."""
        cls.emit_raw(f"{c.Cli.OUTPUT_LOG_LEVEL_DEBUG}: {msg}\n")

    @classmethod
    def header(cls, title: str) -> None:
        """Emit one canonical header block."""
        line = "=" * c.Cli.OUTPUT_HEADER_RULE_WIDTH
        cls.emit_raw(f"\n{line}\n  {title}\n{line}\n")

    @classmethod
    def progress(cls, idx: int, total: int, proj: str, verb: str) -> None:
        """Emit one canonical progress line."""
        width = len(str(total))
        cls.emit_raw(f"[{idx:0{width}d}/{total:0{width}d}] {proj} {verb} ...\n")

    @classmethod
    def status(cls, verb: str, proj: str, *, result: bool, elapsed: float) -> None:
        """Emit one canonical per-item status line."""
        symbol = c.Cli.OUTPUT_STATUS_OK if result else c.Cli.OUTPUT_STATUS_FAIL
        cls.emit_raw(f"  {symbol} {verb:<8} {proj:<24} {elapsed:.2f}s\n")

    @classmethod
    def summary(cls, stats: p.Cli.SummaryStats) -> None:
        """Emit one canonical summary block from a typed summary payload."""
        content = cls.output_summary_content(
            total=stats.total,
            success=stats.success,
            failed=stats.failed,
            skipped=stats.skipped,
        )
        cls.emit_raw(
            f"\n-- {stats.verb} summary --\n{content}  ({stats.elapsed:.2f}s)\n"
        )

    @classmethod
    def gate_result(
        cls, gate: str, count: int, *, passed: bool, elapsed: float
    ) -> None:
        """Emit one canonical gate-result line."""
        symbol = c.Cli.OUTPUT_STATUS_OK if passed else c.Cli.OUTPUT_STATUS_FAIL
        cls.emit_raw(f"    {symbol} {gate:<10} {count:>5} errors  ({elapsed:.2f}s)\n")

    @classmethod
    def project_failure(cls, info: p.Cli.ProjectFailureInfo) -> None:
        """Emit one canonical per-project failure diagnostic from typed info."""
        count_label = (
            f"  [{info.error_count} errors]"
            if info.error_count > 0
            else c.DEFAULT_EMPTY_STRING
        )
        cls.emit_raw(
            f"  {c.Cli.OUTPUT_STATUS_FAIL} {info.project} completed in "
            f"{info.elapsed}s{count_label}  ({info.log_path})\n"
        )
        for line in info.errors[: info.max_show]:
            cls.emit_raw(f"      {line}\n")
        remaining = info.error_count - info.max_show
        if remaining > 0:
            cls.emit_raw(f"      ... and {remaining} more (see log)\n")

    @staticmethod
    def resolve_report_dir(repository_root: Path | str, scope: str, verb: str) -> Path:
        """Resolve standardized report directory path."""
        root_path = (
            Path(repository_root)
            if isinstance(repository_root, str)
            else repository_root
        )
        base = root_path / c.Cli.OUTPUT_REPORTS_DIR_NAME
        if scope == c.Cli.OUTPUT_SCOPE_WORKSPACE:
            return (base / c.Cli.OUTPUT_SCOPE_WORKSPACE / verb).resolve()
        return (base / verb).resolve()

    @classmethod
    def resolve_report_path(
        cls, repository_root: Path | str, scope: str, verb: str, filename: str
    ) -> Path:
        """Resolve standardized report file path."""
        return cls.resolve_report_dir(repository_root, scope, verb) / filename


__all__: list[str] = ["FlextCliUtilitiesOutput"]
