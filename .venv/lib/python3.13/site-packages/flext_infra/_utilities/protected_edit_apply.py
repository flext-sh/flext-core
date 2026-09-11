"""Apply, backup, and pytest flows for protected edit workflows."""

from __future__ import annotations

import ast
import difflib
import operator
import shutil
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from flext_cli import u
from flext_core import r
from flext_infra._utilities.protected_edit_preview import (
    FlextInfraUtilitiesProtectedEditPreview,
)
from flext_infra.constants import c
from flext_infra.models import m
from flext_infra.typings import t

if TYPE_CHECKING:
    from flext_infra.protocols import p


class FlextInfraUtilitiesProtectedEditApply(FlextInfraUtilitiesProtectedEditPreview):
    """Apply, rollback, backup, and pytest helpers for protected edits."""

    @staticmethod
    def _backup_paths_for_updates(
        updates: t.MappingKV[Path, str], *, keep_backup: bool
    ) -> MutableMapping[Path, Path]:
        """Backup paths for updates."""
        if not keep_backup:
            return {}
        backup_paths: MutableMapping[Path, Path] = {}
        for path in updates:
            backup_path = FlextInfraUtilitiesProtectedEditApply._preserve_backup(path)
            if backup_path is not None:
                backup_paths[path] = backup_path
        return backup_paths

    @staticmethod
    def _protected_write_test_failure(
        path: Path,
        request: m.Infra.ProtectedSourceWritesRequest,
        new_errors: t.Infra.LintSnapshot,
    ) -> str | None:
        """Protected write test failure."""
        if new_errors or request.skip_pytest:
            return None
        pytest_result = FlextInfraUtilitiesProtectedEditApply._pytest_failure(
            path, request.workspace
        )
        if pytest_result.failure:
            error_message = pytest_result.error
            return error_message if isinstance(error_message, str) else None
        return None

    @staticmethod
    def _protected_write_reports(
        updates: t.MappingKV[Path, str],
        before_sources: t.MappingKV[Path, str | None],
        before_lints: t.MappingKV[Path, t.Infra.LintSnapshot],
        request: m.Infra.ProtectedSourceWritesRequest,
    ) -> t.Infra.EditResult:
        """Protected write reports."""
        reports: list[str] = []
        failed = False
        after_lints = FlextInfraUtilitiesProtectedEditApply.lint_snapshots(
            tuple(updates), request.workspace, gates=request.gates
        )
        for path in updates:
            new_errors = FlextInfraUtilitiesProtectedEditApply.lint_new_errors(
                before_lints[path], after_lints[path]
            )
            test_fail = (
                FlextInfraUtilitiesProtectedEditApply._protected_write_test_failure(
                    path, request, new_errors
                )
            )
            if not new_errors and not test_fail:
                continue
            failed = True
            reports.extend(
                FlextInfraUtilitiesProtectedEditApply._reverted_report_lines(
                    path,
                    request.workspace,
                    before_sources[path] or "",
                    new_errors,
                    test_fail,
                )
            )
        return (not failed, reports)

    @staticmethod
    def _backup_reports(
        backup_paths: t.MappingKV[Path, Path], workspace: Path
    ) -> list[str]:
        """Backup reports."""
        return [
            "  BACKUP "
            f"{FlextInfraUtilitiesProtectedEditApply._relative_path(path, workspace)}"
            f" -> {backup.name}"
            for path, backup in backup_paths.items()
        ]

    _NO_TESTS_EXIT_CODE = 5
    _NO_TESTS_MARKERS: ClassVar[frozenset[str]] = frozenset({
        "no tests collected",
        "no tests ran",
    })

    @classmethod
    def _has_no_tests_marker(cls, text: str) -> bool:
        """Return whether *text* contains any pytest "no tests" marker."""
        lowered = text.lower()
        return any(marker in lowered for marker in cls._NO_TESTS_MARKERS)

    @classmethod
    def _file_contains_tests(cls, py_file: Path) -> bool:
        """Return whether *py_file* defines pytest-collectable tests."""
        try:
            tree = ast.parse(py_file.read_text(encoding=c.Cli.ENCODING_DEFAULT))
        except SyntaxError:
            return False
        for node in ast.walk(tree):
            if isinstance(
                node, ast.FunctionDef | ast.AsyncFunctionDef
            ) and node.name.startswith("test_"):
                return True
            if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                return True
        return False

    @classmethod
    def _pytest_failure(cls, py_file: Path, workspace: Path) -> p.Result[bool]:
        """Run pytest for a single file and surface a failure message via ``r``."""
        if "tests" not in py_file.parts and not py_file.name.startswith("test_"):
            return r[bool].ok(True)
        if not cls._file_contains_tests(py_file):
            compile_result = u.Cli.run_raw(
                [
                    *cls._workspace_tool_command(workspace, c.Infra.PYTHON),
                    "-m",
                    "py_compile",
                    str(py_file),
                ],
                cwd=cls._command_cwd(py_file, workspace),
                env=cls._command_env(),
                remove_env_keys=cls._COMMAND_ENV_REMOVE_KEYS,
                timeout=c.Infra.TIMEOUT_SHORT,
            )
            if compile_result.failure:
                error = compile_result.error or "py_compile failed"
                return r[bool].fail(error[:300])
            return r[bool].ok(True)
        run_result = u.Cli.run_raw(
            [
                *cls._workspace_tool_command(workspace, "pytest"),
                str(py_file),
                "-x",
                "--tb=short",
                "-q",
            ],
            cwd=cls._command_cwd(py_file, workspace),
            env=cls._command_env(),
            remove_env_keys=cls._COMMAND_ENV_REMOVE_KEYS,
            timeout=c.Infra.TIMEOUT_MEDIUM,
        )
        if run_result.failure:
            error = (run_result.error or "pytest execution failed")[:300]
            return (
                r[bool].ok(True)
                if cls._has_no_tests_marker(error)
                else r[bool].fail(error)
            )
        output = (run_result.value.stdout + run_result.value.stderr)[:300]
        passed_or_no_tests = run_result.value.exit_code == 0 or (
            run_result.value.exit_code == cls._NO_TESTS_EXIT_CODE
            and cls._has_no_tests_marker(output)
        )
        return r[bool].ok(True) if passed_or_no_tests else r[bool].fail(output)

    @staticmethod
    def _preserve_backup(py_file: Path) -> Path | None:
        """Preserve backup."""
        if not py_file.exists():
            return None
        backup_path = py_file.with_suffix(
            py_file.suffix + c.Infra.SAFE_EXECUTION_BAK_SUFFIX
        )
        if not backup_path.exists():
            shutil.copy2(py_file, backup_path)
        return backup_path

    @staticmethod
    def protected_file_edit(
        py_file: Path, *, request: m.Infra.ProtectedFileEditRequest
    ) -> t.Infra.EditResult:
        """Apply one edit, validate lint deltas, and restore on failure."""
        rel = FlextInfraUtilitiesProtectedEditApply._relative_path(
            py_file, request.workspace
        )
        before = FlextInfraUtilitiesProtectedEditApply.lint_snapshot(
            py_file, request.workspace, gates=request.gates
        )
        backup_path = (
            FlextInfraUtilitiesProtectedEditApply._preserve_backup(py_file)
            if request.keep_backup
            else None
        )

        def _restore() -> None:
            """Restore."""
            if request.restore_fn is not None:
                request.restore_fn()
                return
            py_file.write_text(request.before_source, encoding=c.Cli.ENCODING_DEFAULT)

        edit_completed = False
        try:
            request.edit_fn()
            # Canonical normalization: the SAME tool that validates also fixes.
            # Auto-fixable style fallout of a mechanical edit (blank lines,
            # import order, stray pass) is repaired by ruff --fix, never by
            # per-transform hand formatting.
            FlextInfraUtilitiesProtectedEditApply.ruff_fix_files(
                (py_file,), request.workspace
            )
            edit_completed = True
        finally:
            if not edit_completed:
                _restore()

        new_errors = FlextInfraUtilitiesProtectedEditApply.lint_new_errors(
            before,
            FlextInfraUtilitiesProtectedEditApply.lint_snapshot(
                py_file, request.workspace, gates=request.gates
            ),
        )
        test_fail: str | None = (
            None
            if new_errors
            else FlextInfraUtilitiesProtectedEditApply._pytest_failure(
                py_file, request.workspace
            ).fold(on_failure=lambda msg: msg, on_success=lambda _: None)
        )
        if not new_errors and not test_fail:
            if backup_path is None:
                return (True, [])
            return (True, [f"  BACKUP {rel} -> {backup_path.name}"])

        modified = py_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        diff = list(
            difflib.unified_diff(
                request.before_source.splitlines(keepends=True),
                modified.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=3,
            )
        )
        _restore()
        report: t.MutableSequenceOf[str] = [f"  REVERTED {rel}:"]
        report.extend(f"    {line.rstrip()}" for line in diff[:30])
        for tool, messages in new_errors.items():
            report.extend((
                f"    NEW {tool} errors:",
                *(f"      {message}" for message in messages[:5]),
            ))
        if test_fail:
            report.append(f"    pytest failure: {test_fail}")
        return (False, report)

    @staticmethod
    def protected_source_write(
        py_file: Path, *, request: m.Infra.ProtectedSourceWriteRequest
    ) -> t.Infra.EditResult:
        """Write validated source content with protected validation and rollback."""
        original_source = py_file.read_text(encoding=c.Cli.ENCODING_DEFAULT)
        if request.updated_source == original_source:
            return (True, [])

        def _write_updated() -> None:
            """Write updated."""
            py_file.write_text(request.updated_source, encoding=c.Cli.ENCODING_DEFAULT)

        def _restore_original() -> None:
            """Restore original."""
            py_file.write_text(original_source, encoding=c.Cli.ENCODING_DEFAULT)

        return FlextInfraUtilitiesProtectedEditApply.protected_file_edit(
            py_file,
            request=m.Infra.ProtectedFileEditRequest(
                workspace=request.workspace,
                before_source=original_source,
                edit_fn=_write_updated,
                restore_fn=_restore_original,
                keep_backup=request.keep_backup,
                gates=request.gates,
            ),
        )

    @staticmethod
    def protected_source_writes(
        updates: t.MappingKV[Path, str],
        *,
        request: m.Infra.ProtectedSourceWritesRequest,
    ) -> t.Infra.EditResult:
        """Write multiple files transactionally with lint delta validation."""
        if not updates:
            return (True, [])

        normalized_updates = {
            path.resolve(): content
            for path, content in sorted(updates.items(), key=operator.itemgetter(0))
        }
        expected_sources = {
            path.resolve(): content
            for path, content in request.expected_sources.items()
        }
        for path, expected_source in expected_sources.items():
            current_source = (
                path.read_text(encoding=c.Cli.ENCODING_DEFAULT)
                if path.is_file()
                else None
            )
            if current_source != expected_source:
                return (
                    False,
                    [
                        (
                            "CAS failed before protected source writes: "
                            f"source changed for {path}"
                        )
                    ],
                )
        before_sources, before_lints = (
            FlextInfraUtilitiesProtectedEditApply._preview_write_baselines(
                normalized_updates, request.workspace, gates=request.gates
            )
        )
        backup_paths = FlextInfraUtilitiesProtectedEditApply._backup_paths_for_updates(
            normalized_updates, keep_backup=request.keep_backup
        )

        write_completed = False
        try:
            for path, updated_source in normalized_updates.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(updated_source, encoding=c.Cli.ENCODING_DEFAULT)
            if request.post_write is not None:
                request.post_write()
            FlextInfraUtilitiesProtectedEditApply.ruff_fix_files(
                tuple(normalized_updates), request.workspace
            )
            write_completed = True
        finally:
            if not write_completed:
                FlextInfraUtilitiesProtectedEditApply._restore_preview_sources(
                    before_sources
                )

        ok, reports = FlextInfraUtilitiesProtectedEditApply._protected_write_reports(
            normalized_updates, before_sources, before_lints, request
        )
        if not ok:
            FlextInfraUtilitiesProtectedEditApply._restore_preview_sources(
                before_sources
            )
            return (False, reports)

        if not backup_paths:
            return (True, [])
        return (
            True,
            FlextInfraUtilitiesProtectedEditApply._backup_reports(
                backup_paths, request.workspace
            ),
        )


__all__: list[str] = ["FlextInfraUtilitiesProtectedEditApply"]
