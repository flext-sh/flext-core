"""Run flext_infra.refactor CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import FlextInfraRefactorEngine
from .migrate_to_class_mro import FlextInfraRefactorMigrateToClassMRO
from .pydantic_centralizer import FlextInfraRefactorPydanticCentralizer


def main() -> int:
    """Module-level CLI entry point."""
    argv = sys.argv[1:]
    if len(argv) > 0 and argv[0] in {"centralize-pydantic", "centralize-models"}:
        return _run_centralize_pydantic(argv=argv[1:])
    if len(argv) > 0 and argv[0] in {"migrate-to-mro", "migrate-mro"}:
        return _run_migrate_to_mro(argv=argv[1:])
    FlextInfraRefactorEngine.main()
    return 0


def _run_centralize_pydantic(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor centralize-pydantic",
        description="Centralize BaseModel/TypedDict/dict-like aliases into _models.py using AST rewrites",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dry-run", action="store_true", help="Plan only")
    mode.add_argument("--apply", action="store_true", help="Apply migration")
    parser.add_argument(
        "--normalize-remaining",
        action="store_true",
        help="Remove remaining BaseModel/TypedDict bases in non-allowed files",
    )
    args = parser.parse_args(argv)
    apply_changes = bool(args.apply)
    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        args.workspace.resolve(),
        apply_changes=apply_changes,
        normalize_remaining=bool(args.normalize_remaining),
    )
    _ = sys.stdout.write(f"scanned_files={summary['scanned_files']}\n")
    _ = sys.stdout.write(f"touched_files={summary['touched_files']}\n")
    _ = sys.stdout.write(f"moved_classes={summary['moved_classes']}\n")
    _ = sys.stdout.write(f"moved_aliases={summary['moved_aliases']}\n")
    _ = sys.stdout.write(f"normalized_files={summary['normalized_files']}\n")
    _ = sys.stdout.write(f"mode={('apply' if apply_changes else 'dry-run')}\n")
    return 0


def _run_migrate_to_mro(*, argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="flext_infra refactor migrate-mro",
        description="Migrate loose Final/TypeVar/TypeAlias declarations into MRO facade classes and rewrite references",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    parser.add_argument(
        "--target",
        choices=["constants", "all"],
        default="all",
        help="Migration target scope",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dry-run", action="store_true", help="Plan only")
    mode.add_argument("--apply", action="store_true", help="Apply migration")
    args = parser.parse_args(argv)
    service = FlextInfraRefactorMigrateToClassMRO(
        workspace_root=args.workspace.resolve()
    )
    report = service.run(target=args.target, apply_changes=bool(args.apply))
    if len(report.errors) > 0:
        for error in report.errors:
            _ = sys.stderr.write(f"ERROR: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
