"""Generic symbol-rename engine driven by an ``old,new`` CSV list."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from flext_cli import cli, u
from flext_core import r
from flext_infra import m, p, t

_SKIP_DIRS: Final[frozenset[str]] = frozenset({
    "__pycache__",
    "legado",
    "legacy",
    ".git",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
})
_SKIP_SUFFIXES: Final[frozenset[str]] = frozenset({
    ".db",
    ".pyc",
    ".pyo",
    ".so",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".ico",
    ".pdf",
    ".zip",
    ".gz",
    ".whl",
    ".lock",
    ".bak",
})
_ASTGREP: Final[str] = "ast-grep"
_RENAME_COLUMNS: Final[int] = 2


class FlextInfraApplyRenames:
    """Execute prefix-safe, idempotent renames from one CSV source of truth."""

    @staticmethod
    def _pairs(csv_path: Path) -> p.Result[t.SequenceOf[tuple[str, str]]]:
        """Load validated rename pairs longest source name first."""

        def source_name_length(pair: tuple[str, str]) -> int:
            return len(pair[0])

        read_result = u.Cli.files_read_text(csv_path)
        if read_result.failure:
            return r[t.SequenceOf[tuple[str, str]]].fail(
                read_result.error or f"failed to read rename list: {csv_path}"
            )
        rows_result = u.Cli.csv_loads(read_result.value)
        if rows_result.failure:
            return r[t.SequenceOf[tuple[str, str]]].fail(
                rows_result.error or f"failed to parse rename list: {csv_path}"
            )
        rows = rows_result.value
        if not rows or rows[0] != ["old", "new"]:
            return r[t.SequenceOf[tuple[str, str]]].fail(
                f"{csv_path}: header must be exactly 'old,new'"
            )
        pairs: list[tuple[str, str]] = []
        for row in rows[1:]:
            if len(row) != _RENAME_COLUMNS or not row[0].strip() or not row[1].strip():
                return r[t.SequenceOf[tuple[str, str]]].fail(
                    f"{csv_path}: every row must contain non-empty old,new values"
                )
            pairs.append((row[0].strip(), row[1].strip()))
        return r[t.SequenceOf[tuple[str, str]]].ok(
            tuple(sorted(pairs, key=source_name_length, reverse=True))
        )

    @staticmethod
    def _roots(values: t.StrSequence) -> p.Result[t.SequenceOf[Path]]:
        """Resolve existing scan directories from the validated request."""
        roots = tuple(Path(value).resolve() for value in values)
        missing = tuple(path for path in roots if not path.is_dir())
        if missing:
            return r[t.SequenceOf[Path]].fail(
                f"rename root is not a directory: {missing[0]}"
            )
        return r[t.SequenceOf[Path]].ok(roots)

    @staticmethod
    def _text_files(roots: t.SequenceOf[Path]) -> t.SequenceOf[Path]:
        """Collect text candidates while excluding generated and binary trees."""
        files: list[Path] = []
        for root in roots:
            files.extend(
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and not _SKIP_DIRS.intersection(path.parts)
                    and path.suffix not in _SKIP_SUFFIXES
                )
            )
        return tuple(files)

    @staticmethod
    def _scan(
        files: t.SequenceOf[Path], pairs: t.SequenceOf[tuple[str, str]]
    ) -> p.Result[tuple[int, int, t.StrSequence]]:
        """Collect pending occurrences, affected files, and report lines."""
        occurrences = 0
        affected_files = 0
        lines: list[str] = []
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                return r[tuple[int, int, t.StrSequence]].fail(
                    f"failed to read rename target {path}: {exc}"
                )
            file_occurrences = 0
            for old, _new in pairs:
                for match in re.finditer(rf"\b{re.escape(old)}\b", text):
                    line = text.count("\n", 0, match.start()) + 1
                    lines.append(f"{path}:{line}: {old}")
                    occurrences += 1
                    file_occurrences += 1
            affected_files += file_occurrences > 0
        return r[tuple[int, int, t.StrSequence]].ok((
            occurrences,
            affected_files,
            tuple(lines),
        ))

    @staticmethod
    def _apply(
        files: t.SequenceOf[Path],
        roots: t.SequenceOf[Path],
        pairs: t.SequenceOf[tuple[str, str]],
    ) -> p.Result[bool]:
        """Rewrite code nodes first, then remaining text occurrences."""
        root_args = tuple(str(root) for root in roots)
        for old, new in pairs:
            run_result = u.Cli.run_raw((
                _ASTGREP,
                "run",
                "-p",
                old,
                "-r",
                new,
                "-U",
                *root_args,
            ))
            if run_result.failure:
                return r[bool].fail(run_result.error or "ast-grep execution failed")
            output = run_result.value
            if output.exit_code != 0:
                detail = output.stderr.strip() or output.stdout.strip()
                return r[bool].fail(
                    detail or f"ast-grep failed while renaming {old} to {new}"
                )
        for path in files:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                return r[bool].fail(f"failed to read rename target {path}: {exc}")
            updated = text
            for old, new in pairs:
                updated = re.sub(rf"\b{re.escape(old)}\b", new, updated)
            if updated != text:
                write_result = u.Cli.atomic_write_text_file(path, updated)
                if write_result.failure:
                    return r[bool].fail(
                        write_result.error or f"failed to write rename target: {path}"
                    )
        return r[bool].ok(True)

    @classmethod
    def run(
        cls, params: m.Infra.ApplyRenamesInput
    ) -> p.Result[m.Infra.ApplyRenamesReport]:
        """Run a check or apply pass from the canonical request model."""
        csv_path = Path(params.csv).resolve()
        if not csv_path.is_file():
            return r[m.Infra.ApplyRenamesReport].fail(
                f"rename CSV not found: {csv_path}"
            )
        pairs_result = cls._pairs(csv_path)
        if pairs_result.failure:
            return r[m.Infra.ApplyRenamesReport].from_failure(pairs_result)
        roots_result = cls._roots(params.roots)
        if roots_result.failure:
            return r[m.Infra.ApplyRenamesReport].from_failure(roots_result)
        files = cls._text_files(roots_result.value)
        scan_result = cls._scan(files, pairs_result.value)
        if scan_result.failure:
            return r[m.Infra.ApplyRenamesReport].from_failure(scan_result)
        occurrences, affected_files, lines = scan_result.value
        if lines and not params.apply:
            cli.display_text("\n".join(lines))
        if params.apply:
            apply_result = cls._apply(files, roots_result.value, pairs_result.value)
            if apply_result.failure:
                return r[m.Infra.ApplyRenamesReport].from_failure(apply_result)
        report = m.Infra.ApplyRenamesReport(
            label=csv_path.stem,
            files_scanned=len(files),
            occurrences=occurrences if not params.apply else 0,
            files_changed=affected_files if params.apply else 0,
            applied=params.apply,
        )
        cli.display_text(cls.render_text(report))
        return r[m.Infra.ApplyRenamesReport].ok(report)

    @staticmethod
    def render_text(report: m.Infra.ApplyRenamesReport) -> str:
        """Render a concise rename summary for the CLI boundary."""
        if report.applied:
            return (
                f"{report.label}: rewrote {report.files_changed} file(s) "
                f"across {report.files_scanned} scanned"
            )
        return (
            f"{report.label}: {report.occurrences} occurrence(s) "
            f"across {report.files_scanned} file(s)"
        )

    @classmethod
    def execute_command(
        cls, params: m.Infra.ApplyRenamesInput
    ) -> p.Result[t.Cli.ResultValue]:
        """Execute the canonical CLI route and fail on pending check results."""
        result = cls.run(params)
        if result.failure:
            return r[t.Cli.ResultValue].from_failure(result)
        report = result.value
        if not report.applied and report.occurrences:
            return r[t.Cli.ResultValue].fail(
                f"{report.occurrences} pending rename occurrence(s)"
            )
        return r[t.Cli.ResultValue].ok(True)


# This module intentionally exposes NO __main__ entry point. The command is
# already registered canonically in services/cli_routes_refactor.py, bound to
# m.Infra.ApplyRenamesInput and FlextInfraApplyRenames.execute_command, so it
# is reachable through the flext-infra CLI like every other verb.
#
# The argparse block removed here was a second, parallel route to the same
# handler: it re-declared the flags the typed model already owns, so the two
# surfaces could drift apart, and argparse is banned precisely to keep one
# declaration of a command's inputs. Running this file directly is not a
# supported surface.
__all__: list[str] = ["FlextInfraApplyRenames"]
