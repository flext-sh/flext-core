"""Run flext_infra.refactor CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import FlextInfraRefactorEngine
from .migrate_to_class_mro import (
    FlextInfraRefactorMigrateToClassMRO,
)


def main() -> int:
    """Module-level CLI entry point."""
    argv = sys.argv[1:]
    if len(argv) > 0 and argv[0] in {"migrate-to-mro", "migrate-mro"}:
        return _run_migrate_to_mro(argv=argv[1:])
    FlextInfraRefactorEngine.main()
    return 0


def _run_migrate_to_mro(*, argv: list[str]) -> int:

    parser = argparse.ArgumentParser(
        prog="flext_infra refactor migrate-mro",
        description=(
            "Migrate loose Final/TypeVar/TypeAlias declarations "
            "into MRO facade classes and rewrite references"
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root directory (default: cwd)",
    )
    parser.add_argument(
        "--target",
        choices=["constants", "typings", "all"],
        default="all",
        help="Migration target scope",
    )
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dry-run", action="store_true", help="Plan only")
    mode.add_argument("--apply", action="store_true", help="Apply migration")

    args = parser.parse_args(argv)
    service = FlextInfraRefactorMigrateToClassMRO(
        workspace_root=args.workspace.resolve(),
    )
    report = service.run(target=args.target, apply_changes=bool(args.apply))
    print(FlextInfraRefactorMigrateToClassMRO.render_text(report), end="")  # noqa: T201  # JUSTIFIED: CLI text output contract requires stdout printing — https://docs.astral.sh/ruff/rules/print/
    if len(report.errors) > 0:
        for error in report.errors:
            _ = sys.stderr.write(f"ERROR: {error}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
