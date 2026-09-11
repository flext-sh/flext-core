"""FLEXT pyrefly quality gate."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import r
from flext_infra import c, m, t, u
from flext_infra.gates.base_gate import FlextInfraGate

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import p


class FlextInfraPyreflyGate(FlextInfraGate):
    """Pyrefly quality gate."""

    gate_id: ClassVar[str] = c.Infra.PYREFLY
    gate_name: ClassVar[str] = "Pyrefly"
    can_fix: ClassVar[bool] = False
    tool_name: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.PYREFLY][0]
    tool_url: ClassVar[str] = c.Infra.SARIF_TOOL_INFO[c.Infra.PYREFLY][1]

    @override
    def _get_check_dirs(
        self, project_dir: Path, ctx: m.Infra.GateContext
    ) -> t.StrSequence:
        """Check only local Python roots to avoid scanning dependency trees."""
        _ = ctx
        discovered_dirs = u.Infra.discover_python_dirs(project_dir)
        if discovered_dirs:
            return discovered_dirs
        if any(project_dir.glob(c.Infra.EXT_PYTHON_GLOB)) or any(
            project_dir.glob("*.pyi")
        ):
            return ["."]
        return []

    @override
    def _build_check_command(
        self, project_dir: Path, ctx: m.Infra.GateContext, check_dirs: t.StrSequence
    ) -> t.StrSequence:
        """Build check command."""
        json_file = ctx.reports_dir / f"{project_dir.name}-pyrefly.json"
        target_args: t.StrSequence = (
            () if self._has_project_includes_config(project_dir) else tuple(check_dirs)
        )
        return self._python_module_command(
            c.Infra.PYREFLY,
            c.Infra.CHECK,
            *target_args,
            "--config",
            c.Infra.PYPROJECT_FILENAME,
            "--python-interpreter-path",
            sys.executable,
            "--output-format",
            c.Infra.OUTPUT_JSON,
            "-o",
            str(json_file),
            "--summary=none",
        )

    @override
    def _parse_check_output(
        self, result: p.Cli.CommandOutput, project_dir: Path, ctx: m.Infra.GateContext
    ) -> tuple[bool, t.SequenceOf[m.Infra.Issue]]:
        """Parse check output."""
        json_file = ctx.reports_dir / f"{project_dir.name}-pyrefly.json"
        issues: t.MutableSequenceOf[m.Infra.Issue] = []
        if json_file.exists():
            read = u.Cli.files_read_json(json_file)
            if read.failure:
                issues.append(
                    m.Infra.Issue(
                        file="<pyrefly-output>",
                        line=0,
                        column=0,
                        code="PARSE_ERROR",
                        message=f"pyrefly output unreadable/invalid: {read.error}",
                        severity="ERROR",
                    )
                )
                return False, issues
            parsed_value = read.value
            error_items = self._error_items_from_output(parsed_value)
            if error_items.failure:
                issues.append(
                    self._parse_error_issue(
                        error_items.error or "Tool output parsing failed"
                    )
                )
                return False, issues
            try:
                issues.extend(self._issues_from_error_items(error_items.value))
            except c.EXC_VALIDATION_TYPE as err:
                issues.append(
                    m.Infra.Issue(
                        file="<pyrefly-output>",
                        line=0,
                        column=0,
                        code="PARSE_ERROR",
                        message=(
                            "Tool output parsing failed while collecting diagnostics: "
                            f"{type(err).__name__}"
                        ),
                        severity=c.Infra.ERROR,
                    )
                )
        if (not issues) and result.exit_code != 0:
            message = (result.stderr or result.stdout).strip()
            if not message:
                message = (
                    f"pyrefly exited with code {result.exit_code} "
                    "without JSON diagnostics"
                )
            issues.append(
                m.Infra.Issue(
                    file=c.Infra.PYPROJECT_FILENAME,
                    line=1,
                    column=1,
                    code="pyrefly-exec",
                    message=message,
                    severity=c.Infra.ERROR,
                )
            )
        return result.exit_code == 0, issues

    @staticmethod
    def _parse_error_issue(message: str) -> m.Infra.Issue:
        """Return the canonical parse-error issue."""
        return m.Infra.Issue(
            file="<pyrefly-output>",
            line=0,
            column=0,
            code="PARSE_ERROR",
            message=message,
            severity=c.Infra.ERROR,
        )

    @classmethod
    def _error_items_from_output(
        cls, parsed_value: t.Infra.InfraValue
    ) -> p.Result[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]]:
        """Return pyrefly error items from either object or list JSON output."""
        if isinstance(parsed_value, Mapping):
            return cls._error_items_from_mapping(parsed_value)
        if isinstance(parsed_value, list):
            return cls._error_items_from_list(parsed_value)
        return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].ok(())

    @staticmethod
    def _error_items_from_mapping(
        parsed_value: t.Infra.InfraValue,
    ) -> p.Result[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]]:
        """Return error items from object-shaped pyrefly JSON output."""
        try:
            parsed_mapping = u.Cli.json_as_mapping(parsed_value)
        except c.EXC_VALIDATION_TYPE as err:
            return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].fail(
                f"Tool output parsing failed: {type(err).__name__}"
            )
        try:
            error_items = u.Cli.json_deep_mapping_list(
                parsed_mapping, c.Infra.PYREFLY_ERRORS_KEY
            )
        except c.EXC_VALIDATION_TYPE as err:
            return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].fail(
                f"Tool output parsing failed: {type(err).__name__}"
            )
        return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].ok(error_items)

    @staticmethod
    def _error_items_from_list(
        parsed_value: t.Infra.InfraValue,
    ) -> p.Result[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]]:
        """Return error items from list-shaped pyrefly JSON output."""
        try:
            error_items = u.Cli.json_as_mapping_list(parsed_value)
        except c.EXC_VALIDATION_TYPE as err:
            return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].fail(
                f"Tool output parsing failed: {type(err).__name__}"
            )
        return r[t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]]].ok(error_items)

    @staticmethod
    def _issues_from_error_items(
        error_items: t.SequenceOf[t.MappingKV[str, t.Infra.InfraValue]],
    ) -> t.SequenceOf[m.Infra.Issue]:
        """Convert validated pyrefly error items to gate issues."""
        return tuple(
            m.Infra.Issue(
                file=u.Cli.json_pick_str(err, "path", "?"),
                line=u.Cli.json_pick_int(err, "line"),
                column=u.Cli.json_pick_int(err, "column"),
                code=u.Cli.json_pick_str(err, "name"),
                message=u.Cli.json_pick_str(err, "description"),
                severity=u.Cli.json_pick_str(err, "severity", c.Infra.ERROR),
            )
            for err in error_items
            if "/.venv/" not in u.Cli.json_pick_str(err, "path", "")
            and "/site-packages/" not in u.Cli.json_pick_str(err, "path", "")
        )

    @staticmethod
    def _has_project_includes_config(project_dir: Path) -> bool:
        """Return whether pyproject.toml declares pyrefly project-includes."""
        doc = u.Cli.toml_read(project_dir / c.Infra.PYPROJECT_FILENAME)
        if doc is None:
            return False
        tool_table = u.Cli.toml_table_child(doc, c.Infra.TOOL)
        pyrefly_table = (
            None
            if tool_table is None
            else u.Cli.toml_table_child(tool_table, c.Infra.PYREFLY)
        )
        return (
            pyrefly_table is not None
            and u.Cli.toml_item_child(pyrefly_table, c.Infra.PROJECT_INCLUDES)
            is not None
        )


__all__: list[str] = ["FlextInfraPyreflyGate"]
