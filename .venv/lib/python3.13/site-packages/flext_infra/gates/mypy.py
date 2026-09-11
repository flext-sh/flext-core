"""FLEXT mypy quality gate."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar, override

from flext_infra import c, m, t, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraMypyGate(FlextInfraGate):
    """Gate for Mypy type checking."""

    gate_id: ClassVar[str] = c.Infra.MYPY
    gate_name: ClassVar[str] = "Mypy"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.MYPY][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.MYPY][1]

    @staticmethod
    def _config_exclude(config_path: Path) -> re.Pattern[str] | None:
        """Return the compiled [tool.mypy].exclude regex, if configured."""
        doc = u.Cli.toml_read(config_path)
        if doc is None:
            return None
        tool_table = u.Cli.toml_table_child(doc, c.Infra.TOOL)
        if tool_table is None:
            return None
        mypy_table = u.Cli.toml_table_child(tool_table, c.Infra.MYPY)
        if mypy_table is None:
            return None
        exclude = mypy_table.get("exclude")
        if not isinstance(exclude, str) or not exclude:
            return None
        return re.compile(exclude)

    @staticmethod
    def _has_real_module(directory: Path) -> bool:
        """True when the dir holds a Python module other than __init__.py."""
        for py_file in directory.rglob(c.Infra.EXT_PYTHON_GLOB):
            if py_file.name != c.Infra.INIT_PY:
                return True
        return any(directory.rglob("*.pyi"))

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Check local Python roots directly instead of recursively scanning ``.``."""
        # NOTE (multi-agent): Mypy also owns typed tests, including PEP 420 test
        # roots without __init__.py. Empty or explicitly excluded roots are
        # still omitted because Mypy aborts when they are passed positionally.
        exclude = self._config_exclude(self._resolve_config(project_dir, ctx))
        discovered_dirs = [
            directory
            for directory in self._dirs_with_py(
                project_dir, (*c.Infra.CHECK_DIRS_REPOSITORY, c.Infra.DIR_TESTS)
            )
            if self._has_real_module(project_dir / directory)
            and (exclude is None or not exclude.match(f"{directory}/"))
        ]
        root_files = [
            path.name
            for path in sorted(project_dir.iterdir())
            if path.is_file() and path.suffix in {".py", ".pyi"}
        ]
        if discovered_dirs or root_files:
            return [*discovered_dirs, *root_files]
        return []

    def _resolve_config(self, project_dir: Path, ctx: m.Infra.GateContext) -> Path:
        """Resolve Mypy settings from the project, then the workspace."""
        pyproject_name: str = c.Infra.PYPROJECT_FILENAME
        proj_py = project_dir / pyproject_name
        doc = u.Cli.toml_read(proj_py)
        if doc is not None:
            tool_table = u.Cli.toml_table_child(doc, c.Infra.TOOL)
            if (
                tool_table is not None
                and u.Cli.toml_table_child(tool_table, c.Infra.MYPY) is not None
            ):
                return proj_py
        repository_root: Path = ctx.repository_root
        return repository_root / pyproject_name

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build check command."""
        cfg = self._resolve_config(project_dir, ctx)
        return u.Infra.mypy_limited_command(
            self._python_module_command(
                c.Infra.MYPY,
                *check_dirs,
                "--config-file",
                str(cfg),
                "--output",
                c.Infra.OUTPUT_JSON,
            )
        )

    @override
    def _check_timeout(self, project_dir: Path, ctx: m.Infra.GateContext) -> int:
        """Keep the outer runner alive through the controlled Mypy deadline."""
        _ = project_dir, ctx
        return u.Infra.mypy_runner_timeout()

    @override
    def _check_env(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrMapping | None:
        """Check env."""
        _ = project_dir
        typings_generated = ctx.repository_root / c.Infra.DIR_TYPINGS / "generated"
        if not typings_generated.is_dir():
            return None
        base_env = u.Cli.process_env()
        existing = base_env.get("MYPYPATH", "")
        mypy_path = str(typings_generated) + (f":{existing}" if existing else "")
        return u.Cli.process_env(overrides={"MYPYPATH": mypy_path})

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse check output."""
        _ = project_dir, ctx
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        if resource_diagnostic := u.Infra.mypy_failure_diagnostic(result):
            return (
                False,
                (
                    m.Infra.Issue(
                        file=c.Infra.PYPROJECT_FILENAME,
                        line=1,
                        column=1,
                        code="mypy-resource-limit",
                        message=resource_diagnostic,
                        severity=c.Infra.ERROR,
                    ),
                ),
            )
        for raw_line in (result.stdout or "").splitlines():
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                line_data: t.MappingKV[str, t.Infra.InfraValue] = (
                    t.Infra.INFRA_MAPPING_ADAPTER.validate_json(stripped)
                )
            except c.ValidationError:
                continue
            try:
                severity = u.Cli.json_pick_str(line_data, "severity", c.Infra.ERROR)
                if severity in c.Infra.VALID_GATE_SEVERITIES:
                    issues.append(
                        m.Infra.Issue(
                            file=u.Cli.json_pick_str(line_data, "file", "?"),
                            line=u.Cli.json_pick_int(line_data, "line"),
                            column=u.Cli.json_pick_int(line_data, "column"),
                            code=u.Cli.json_pick_str(line_data, "code"),
                            message=u.Cli.json_pick_str(line_data, "message"),
                            severity=severity,
                        )
                    )
            except c.ValidationError:
                continue
        if (not issues) and result.exit_code != 0:
            message = (result.stderr or result.stdout).strip()
            if not message:
                message = (
                    f"mypy exited with code {result.exit_code} without JSON diagnostics"
                )
            issues.append(
                m.Infra.Issue(
                    file=c.Infra.PYPROJECT_FILENAME,
                    line=1,
                    column=1,
                    code="mypy-exec",
                    message=message,
                    severity=c.Infra.ERROR,
                )
            )
        return result.exit_code == 0, issues


__all__: list[str] = ["FlextInfraMypyGate"]
