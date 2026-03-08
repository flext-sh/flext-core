"""Run flext_infra.refactor CLI."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from .migrate_to_class_mro import FlextInfraRefactorMigrateToClassMRO
from .namespace_enforcer import FlextInfraNamespaceEnforcer
from .pydantic_centralizer import FlextInfraRefactorPydanticCentralizer


def main() -> int:
    """Module-level CLI entry point."""
    argv = sys.argv[1:]
    if len(argv) > 0 and argv[0] in {"centralize-pydantic", "centralize-models"}:
        return _run_centralize_pydantic(argv=argv[1:])
    if len(argv) > 0 and argv[0] in {"migrate-to-mro", "migrate-mro"}:
        return _run_migrate_to_mro(argv=argv[1:])
    if len(argv) > 0 and argv[0] in {"namespace-enforce", "enforce-namespaces"}:
        return _run_namespace_enforce(argv=argv[1:])
    if len(argv) > 0 and argv[0] in {"ultrawork-models", "ultrawork"}:
        return _run_ultrawork_models(argv=argv[1:])
    _ = sys.stderr.write(
        "Usage: python -m flext_infra.refactor [centralize-pydantic|migrate-mro|namespace-enforce|ultrawork-models] ...\n"
    )
    return 2


def _copy_workspace_for_dry_run(workspace: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="flext-refactor-dryrun-"))
    workspace_copy = temp_root / "workspace"

    def _ignore(_dir: str, names: list[str]) -> set[str]:
        ignored: set[str] = set()
        for name in names:
            if name in {".git", ".beads", "__pycache__"}:
                ignored.add(name)
                continue
            if name.endswith(".sock"):
                ignored.add(name)
        return ignored

    _ = shutil.copytree(workspace, workspace_copy, ignore=_ignore)
    source_venv_cfg = workspace / ".venv" / "pyvenv.cfg"
    if source_venv_cfg.exists():
        target_venv = workspace_copy / ".venv"
        target_venv.mkdir(parents=True, exist_ok=True)
        _ = shutil.copy2(source_venv_cfg, target_venv / "pyvenv.cfg")
    return workspace_copy


def _run_centralize_pydantic(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor centralize-pydantic",
        description="Centralize BaseModel/TypedDict/dict-like aliases into _models.py using AST rewrites",
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    _ = mode.add_argument("--dry-run", action="store_true", help="Plan only")
    _ = mode.add_argument("--apply", action="store_true", help="Apply migration")
    _ = parser.add_argument(
        "--normalize-remaining",
        action="store_true",
        help="Remove remaining BaseModel/TypedDict bases in non-allowed files",
    )
    _ = parser.add_argument(
        "--dry-run-copy-workspace",
        action="store_true",
        help="Run dry-run against a temporary full workspace copy",
    )
    args = parser.parse_args(argv)
    apply_changes = bool(args.apply)
    workspace_path = args.workspace.resolve()
    if bool(args.dry_run_copy_workspace):
        workspace_path = _copy_workspace_for_dry_run(workspace_path)
        apply_changes = False
    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        workspace_path,
        apply_changes=apply_changes,
        normalize_remaining=bool(args.normalize_remaining),
    )
    _ = sys.stdout.write(f"scanned_files={summary['scanned_files']}\n")
    _ = sys.stdout.write(f"touched_files={summary['touched_files']}\n")
    _ = sys.stdout.write(f"moved_classes={summary['moved_classes']}\n")
    _ = sys.stdout.write(f"moved_aliases={summary['moved_aliases']}\n")
    _ = sys.stdout.write(f"normalized_files={summary['normalized_files']}\n")
    _ = sys.stdout.write(
        f"detected_model_violations={summary['detected_model_violations']}\n"
    )
    _ = sys.stdout.write(
        f"detected_alias_violations={summary['detected_alias_violations']}\n"
    )
    _ = sys.stdout.write(f"created_model_files={summary['created_model_files']}\n")
    _ = sys.stdout.write(f"parse_syntax_errors={summary['parse_syntax_errors']}\n")
    _ = sys.stdout.write(f"parse_encoding_errors={summary['parse_encoding_errors']}\n")
    _ = sys.stdout.write(f"parse_io_errors={summary['parse_io_errors']}\n")
    _ = sys.stdout.write(f"created_typings_files={summary['created_typings_files']}\n")
    _ = sys.stdout.write(
        f"skipped_non_necessary_apply={summary.get('skipped_non_necessary_apply', 0)}\n"
    )
    _ = sys.stdout.write(
        f"skipped_nonpackage_apply={summary['skipped_nonpackage_apply']}\n"
    )
    _ = sys.stdout.write(f"workspace={workspace_path}\n")
    _ = sys.stdout.write(f"mode={('apply' if apply_changes else 'dry-run')}\n")
    return 0


def _run_migrate_to_mro(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor migrate-mro",
        description="Migrate loose Final/TypeVar/TypeAlias declarations into MRO facade classes and rewrite references",
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    _ = parser.add_argument(
        "--target",
        choices=["constants", "typings", "protocols", "models", "utilities", "all"],
        default="all",
        help="Migration target scope",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    _ = mode.add_argument("--dry-run", action="store_true", help="Plan only")
    _ = mode.add_argument("--apply", action="store_true", help="Apply migration")
    _ = parser.add_argument(
        "--dry-run-copy-workspace",
        action="store_true",
        help="Run dry-run against a temporary full workspace copy",
    )
    args = parser.parse_args(argv)
    workspace_path = args.workspace.resolve()
    apply_changes = bool(args.apply)
    if bool(args.dry_run_copy_workspace):
        workspace_path = _copy_workspace_for_dry_run(workspace_path)
        apply_changes = False
    service = FlextInfraRefactorMigrateToClassMRO(workspace_root=workspace_path)
    report = service.run(target=args.target, apply_changes=apply_changes)
    _ = sys.stdout.write(FlextInfraRefactorMigrateToClassMRO.render_text(report))
    if len(report.errors) > 0:
        for error in report.errors:
            _ = sys.stderr.write(f"ERROR: {error}\n")
        return 1
    return 0


def _run_namespace_enforce(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor namespace-enforce",
        description="Scan workspace for namespace violations: missing facades, loose objects, import violations, cyclic imports",
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    _ = mode.add_argument("--dry-run", action="store_true", help="Scan only (default)")
    _ = mode.add_argument("--apply", action="store_true", help="Apply auto-fixes")
    _ = parser.add_argument(
        "--dry-run-copy-workspace",
        action="store_true",
        help="Run against a temporary full workspace copy",
    )
    args = parser.parse_args(argv)
    workspace_path = args.workspace.resolve()
    apply_changes = bool(args.apply)
    if bool(args.dry_run_copy_workspace):
        workspace_path = _copy_workspace_for_dry_run(workspace_path)
        # When --apply was explicitly given, apply to the copy (safe test)
        # Otherwise default to scan-only on the copy
        if not apply_changes:
            apply_changes = False
    enforcer = FlextInfraNamespaceEnforcer(workspace_root=workspace_path)
    report = enforcer.enforce(apply_changes=apply_changes)
    _ = sys.stdout.write(FlextInfraNamespaceEnforcer.render_text(report))
    has_violations = (
        report.total_facades_missing > 0
        or report.total_loose_objects > 0
        or report.total_import_violations > 0
        or report.total_internal_import_violations > 0
        or report.total_runtime_alias_violations > 0
        or report.total_manual_protocol_violations > 0
        or report.total_manual_typing_violations > 0
        or report.total_compatibility_alias_violations > 0
        or report.total_future_violations > 0
    )
    if has_violations:
        return 1
    return 0


def _run_ultrawork_models(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor ultrawork-models",
        description="Run full AST model centralization + MRO + namespace enforcement workflow",
    )
    _ = parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    _ = mode.add_argument("--dry-run", action="store_true", help="Plan only")
    _ = mode.add_argument("--apply", action="store_true", help="Apply migration")
    _ = parser.add_argument(
        "--normalize-remaining",
        action="store_true",
        help="Remove remaining BaseModel/TypedDict bases in non-allowed files",
    )
    _ = parser.add_argument(
        "--dry-run-copy-workspace",
        action="store_true",
        help="Run against a temporary full workspace copy",
    )
    args = parser.parse_args(argv)
    workspace_path = args.workspace.resolve()
    apply_changes = bool(args.apply)
    if bool(args.dry_run_copy_workspace):
        workspace_path = _copy_workspace_for_dry_run(workspace_path)
        apply_changes = False
    centralize_summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        workspace_path,
        apply_changes=apply_changes,
        normalize_remaining=bool(args.normalize_remaining),
    )
    mro_report = FlextInfraRefactorMigrateToClassMRO(workspace_root=workspace_path).run(
        target="all",
        apply_changes=apply_changes,
    )
    namespace_report = FlextInfraNamespaceEnforcer(
        workspace_root=workspace_path
    ).enforce(apply_changes=apply_changes)
    _ = sys.stdout.write(f"workspace={workspace_path}\n")
    _ = sys.stdout.write(f"mode={('apply' if apply_changes else 'dry-run')}\n")
    _ = sys.stdout.write(f"scanned_files={centralize_summary['scanned_files']}\n")
    _ = sys.stdout.write(f"touched_files={centralize_summary['touched_files']}\n")
    _ = sys.stdout.write(
        f"detected_model_violations={centralize_summary['detected_model_violations']}\n"
    )
    _ = sys.stdout.write(
        f"detected_alias_violations={centralize_summary['detected_alias_violations']}\n"
    )
    _ = sys.stdout.write(f"moved_classes={centralize_summary['moved_classes']}\n")
    _ = sys.stdout.write(f"moved_aliases={centralize_summary['moved_aliases']}\n")
    _ = sys.stdout.write(
        f"created_model_files={centralize_summary['created_model_files']}\n"
    )
    _ = sys.stdout.write(
        f"parse_syntax_errors={centralize_summary['parse_syntax_errors']}\n"
    )
    _ = sys.stdout.write(
        f"parse_encoding_errors={centralize_summary['parse_encoding_errors']}\n"
    )
    _ = sys.stdout.write(f"parse_io_errors={centralize_summary['parse_io_errors']}\n")
    _ = sys.stdout.write(
        f"created_typings_files={centralize_summary['created_typings_files']}\n"
    )
    _ = sys.stdout.write(
        "skipped_non_necessary_apply="
        f"{centralize_summary.get('skipped_non_necessary_apply', 0)}\n"
    )
    _ = sys.stdout.write(
        f"skipped_nonpackage_apply={centralize_summary['skipped_nonpackage_apply']}\n"
    )
    _ = sys.stdout.write(
        f"mro_remaining_violations={mro_report.remaining_violations}\n"
    )
    _ = sys.stdout.write(f"mro_failures={mro_report.mro_failures}\n")
    _ = sys.stdout.write(
        f"namespace_missing_facades={namespace_report.total_facades_missing}\n"
    )
    _ = sys.stdout.write(
        f"namespace_loose_objects={namespace_report.total_loose_objects}\n"
    )
    _ = sys.stdout.write(
        f"namespace_import_violations={namespace_report.total_import_violations}\n"
    )
    _ = sys.stdout.write(
        f"namespace_cyclic_imports={namespace_report.total_cyclic_imports}\n"
    )
    _ = sys.stdout.write(
        "namespace_runtime_alias_violations="
        f"{namespace_report.total_runtime_alias_violations}\n"
    )
    _ = sys.stdout.write(
        f"namespace_missing_future={namespace_report.total_future_violations}\n"
    )
    _ = sys.stdout.write(
        "namespace_manual_protocols="
        f"{namespace_report.total_manual_protocol_violations}\n"
    )
    _ = sys.stdout.write(
        "namespace_manual_typing_aliases="
        f"{namespace_report.total_manual_typing_violations}\n"
    )
    _ = sys.stdout.write(
        "namespace_compatibility_aliases="
        f"{namespace_report.total_compatibility_alias_violations}\n"
    )
    if len(mro_report.errors) > 0:
        for error in mro_report.errors:
            _ = sys.stderr.write(f"ERROR: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
