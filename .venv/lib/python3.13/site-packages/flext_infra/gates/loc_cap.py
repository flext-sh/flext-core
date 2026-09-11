"""FLEXT module-cap SUPREME LAW (§3.1) quality gate.

Enforces the per-module logical-LOC ceiling using scc's code-line count.
Per-class / per-method / per-function caps require AST and are out of scope
for this tool-driven gate (scc reports at file granularity only).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraLocCapGate(FlextInfraGate):
    """Flag any module whose scc `Code` LOC exceeds ``c.Infra.LOC_CAP_MAX``."""

    gate_id: ClassVar[str] = "loc-cap"
    gate_name: ClassVar[str] = "MODULE-LOC SUPREME LAW"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["loc-cap"][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO["loc-cap"][1]

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Run scc over the project's Python directories, emitting per-file JSON."""
        _ = project_dir, ctx
        return [c.Infra.SCC_BINARY, "--format", "json", "--by-file", *check_dirs]

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse scc JSON into one Issue per over-cap module."""
        _ = project_dir, ctx
        if result.exit_code != 0:
            return (
                False,
                (
                    m.Infra.Issue(
                        file="<scc>",
                        line=0,
                        column=0,
                        code="LOC_CAP_EXEC",
                        message=result.stderr or "scc execution failed",
                        severity="ERROR",
                    ),
                ),
            )
        issues = self._files_over_cap(result.stdout or "[]", c.Infra.LOC_CAP_MAX)
        return len(issues) == 0, issues

    @staticmethod
    def _file_code_line(file_entry: t.JsonValue) -> tuple[str, int] | None:
        """Return one scc by-file entry's ``(path, code)`` pair, or ``None``."""
        if not isinstance(file_entry, Mapping):
            return None
        code = u.Cli.json_pick_int(file_entry, "Code")
        name = u.Cli.json_pick_str(file_entry, "Location", "?")
        return name, code

    @classmethod
    def _python_language_files(
        cls, language_entry: t.JsonValue
    ) -> t.SequenceOf[tuple[str, int]]:
        """Return every ``(path, code)`` pair from one scc language entry.

        Yields nothing for a non-Python entry or one carrying no file list.
        """
        if not isinstance(language_entry, Mapping):
            return ()
        if language_entry.get("Name") != c.Infra.SCC_PYTHON_LANG:
            return ()
        files = language_entry.get("Files")
        if not isinstance(files, list):
            return ()
        picked = (cls._file_code_line(file_entry) for file_entry in files)
        return tuple(pair for pair in picked if pair is not None)

    @classmethod
    def _python_file_code_lines(
        cls, data: t.JsonValue
    ) -> t.SequenceOf[tuple[str, int]]:
        """Return every Python file's ``(path, code)`` pair across an scc payload."""
        if not isinstance(data, list):
            return ()
        pairs: t.MutableSequenceOf[tuple[str, int]] = []
        for language_entry in data:
            pairs.extend(cls._python_language_files(language_entry))
        return tuple(pairs)

    @staticmethod
    def _issue_for_over_cap(path: str, code: int, cap: int) -> m.Infra.Issue:
        """Build the ``LOC_CAP`` issue for one module past the SUPREME LAW cap."""
        return m.Infra.Issue(
            file=path,
            line=code,
            column=0,
            code="LOC_CAP",
            message=f"{code} code LOC exceeds {cap}-line SUPREME LAW",
            severity="ERROR",
        )

    @classmethod
    def _files_over_cap(cls, scc_json: str, cap: int) -> tuple[m.Infra.Issue, ...]:
        """Extract over-cap modules from an `scc --format json --by-file` payload.

        Pure function (no subprocess) so the cap logic is unit-testable against
        a literal scc fixture.
        """
        parsed = u.Cli.json_parse(scc_json or "[]")
        empty: t.JsonValue = []
        data = parsed.unwrap() if parsed.success else empty
        return tuple(
            cls._issue_for_over_cap(path, code, cap)
            for path, code in cls._python_file_code_lines(data)
            if code > cap
        )


__all__: list[str] = ["FlextInfraLocCapGate"]
