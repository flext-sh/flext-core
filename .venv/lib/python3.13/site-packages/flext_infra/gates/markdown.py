"""FLEXT markdown quality gate."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p, t


class FlextInfraMarkdownGate(FlextInfraGate):
    """Markdown quality gate."""

    gate_id: ClassVar[str] = c.Infra.MARKDOWN
    gate_name: ClassVar[str] = "Markdown"
    # flext-38p39: the linter flags MD009/MD012 and friends with its own `[*]`
    # auto-fixable marker, so `make check` blocked on findings that no canonical
    # verb could repair -- `make fmt APPLY=Y` covers Python only and `make fix
    # APPLY=Y` skipped this gate, both exiting 0. The tool supports `--fix`, so
    # the gate offers it and the canonical sequence can reach green.
    can_fix: ClassVar[bool] = True
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.MARKDOWN][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.MARKDOWN][1]

    def _collect_markdown_files(self, project_dir: Path) -> t.SequenceOf[Path]:
        """Collect markdown files."""
        markdown_files: list[Path] = []
        for path in u.Infra.iter_matching_files(project_dir, includes=["*.md"]):
            relative_parts = path.relative_to(project_dir).parts
            if any(part in c.Infra.CHECK_EXCLUDED_DIRS for part in relative_parts):
                continue
            if (
                len(relative_parts) > 1
                and relative_parts[0] == ".github"
                and relative_parts[1] in c.Infra.GITHUB_AGENT_PROJECTION_DIRS
            ):
                continue
            markdown_files.append(path)
        return markdown_files

    def _resolve_config_args(self, project_dir: Path) -> t.StrSequence:
        """Resolve only the repository-local markdown settings owner."""
        config_path = project_dir / ".markdownlint.json"
        if not config_path.is_file():
            return ["--no-config"]
        return ["--config", str(config_path.resolve())]

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Return relative markdown file paths (doubles as check_dirs for _build_check_command)."""
        _ = ctx
        return [
            str(path.relative_to(project_dir))
            for path in self._collect_markdown_files(project_dir)
        ]

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build check command."""
        _ = ctx
        return self._python_console_script_command(
            c.Infra.RUMDL,
            "check",
            "--no-cache",
            "--color",
            "never",
            "--output-format",
            "text",
            "--deny-config-warnings",
            *self._resolve_config_args(project_dir),
            *check_dirs,
        )

    @override
    def _build_fix_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, targets: t.StrSequence
    ) -> t.StrSequence:
        """Build the fix command from the tool's FORMATTER, not its linter.

        ``rumdl check --fix`` is a linter: it exits non-zero whenever a finding
        has no autofix, so a run that repaired every fixable file still failed
        the verb and `make fix APPLY=Y` could never reach green. ``rumdl fmt``
        applies the same fixes with formatter-style exit codes, which is the
        contract the mutating verb promises. It accepts neither
        ``--output-format`` nor ``--deny-config-warnings`` (both are check-only
        reporting flags), so the fix surface carries only what it defines.
        """
        _ = ctx
        return self._python_console_script_command(
            c.Infra.RUMDL,
            "fmt",
            "--no-cache",
            "--color",
            "never",
            *self._resolve_config_args(project_dir),
            *targets,
        )

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse check output."""
        _ = ctx
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        for line in (result.stdout + "\n" + result.stderr).splitlines():
            match = c.Infra.MARKDOWN_RE.match(line.strip())
            if not match:
                continue
            issues.append(
                m.Infra.Issue(
                    file=match.group("file"),
                    line=int(match.group("line")),
                    column=int(match.group("col") or 1),
                    code=match.group("code"),
                    message=match.group("msg"),
                )
            )
        if result.exit_code != 0 and not issues:
            detail = (result.stderr or result.stdout).strip() or "no diagnostics"
            issues.append(
                m.Infra.Issue(
                    file=str(project_dir),
                    line=1,
                    column=1,
                    code="TOOL_ERROR",
                    message=f"rumdl exited with code {result.exit_code}: {detail}",
                    severity="ERROR",
                )
            )
        return result.exit_code == 0, issues


__all__: list[str] = ["FlextInfraMarkdownGate"]
