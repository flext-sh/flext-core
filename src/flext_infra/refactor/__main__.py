"""Run flext_infra.refactor CLI."""

from __future__ import annotations

import sys
from pathlib import Path

from flext_infra import output, u

from .census import FlextInfraRefactorCensus
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
    if len(argv) > 0 and argv[0] in {"census", "utilities-census"}:
        return _run_census(argv=argv[1:])
    output.error(
        "Usage: python -m flext_infra.refactor "
        "[centralize-pydantic|migrate-mro|namespace-enforce|ultrawork-models|census] ..."
    )
    return 2


def _run_centralize_pydantic(*, argv: list[str]) -> int:
    parser = u.Infra.create_refactor_parser(
        prog="flext_infra refactor centralize-pydantic",
        description="Centralize BaseModel/TypedDict/dict-like aliases into _models.py using AST rewrites",
    )
    _ = parser.add_argument(
        "--normalize-remaining",
        action="store_true",
        help="Remove remaining BaseModel/TypedDict bases in non-allowed files",
    )
    args = parser.parse_args(argv)
    workspace_path, apply_changes = u.Infra.resolve_workspace_args(args)
    summary = FlextInfraRefactorPydanticCentralizer.centralize_workspace(
        workspace_path,
        apply_changes=apply_changes,
        normalize_remaining=bool(args.normalize_remaining),
    )
    output.metrics(
        {"workspace": workspace_path, "mode": "apply" if apply_changes else "dry-run"},
        summary,
    )
    return 0


def _run_migrate_to_mro(*, argv: list[str]) -> int:
    parser = u.Infra.create_refactor_parser(
        prog="flext_infra refactor migrate-mro",
        description="Migrate loose Final/TypeVar/TypeAlias declarations into MRO facade classes and rewrite references",
    )
    _ = parser.add_argument(
        "--target",
        choices=["constants", "typings", "protocols", "models", "utilities", "all"],
        default="all",
        help="Migration target scope",
    )
    args = parser.parse_args(argv)
    workspace_path, apply_changes = u.Infra.resolve_workspace_args(args)
    service = FlextInfraRefactorMigrateToClassMRO(workspace_root=workspace_path)
    report = service.run(target=args.target, apply_changes=apply_changes)
    output.write(FlextInfraRefactorMigrateToClassMRO.render_text(report))
    if len(report.errors) > 0:
        for error in report.errors:
            output.error(error)
        return 1
    return 0


def _run_namespace_enforce(*, argv: list[str]) -> int:
    parser = u.Infra.create_refactor_parser(
        prog="flext_infra refactor namespace-enforce",
        description="Scan workspace for namespace violations: missing facades, loose objects, import violations, cyclic imports",
    )
    args = parser.parse_args(argv)
    workspace_path, apply_changes = u.Infra.resolve_workspace_args(args)
    enforcer = FlextInfraNamespaceEnforcer(workspace_root=workspace_path)
    report = enforcer.enforce(apply_changes=apply_changes)
    sys.stdout.write(FlextInfraNamespaceEnforcer.render_text(report))
    sys.stdout.flush()
    if report.has_violations:
        return 1
    return 0


def _run_ultrawork_models(*, argv: list[str]) -> int:
    parser = u.Infra.create_refactor_parser(
        prog="flext_infra refactor ultrawork-models",
        description="Run full AST model centralization + MRO + namespace enforcement workflow",
    )
    _ = parser.add_argument(
        "--normalize-remaining",
        action="store_true",
        help="Remove remaining BaseModel/TypedDict bases in non-allowed files",
    )
    args = parser.parse_args(argv)
    workspace_path, apply_changes = u.Infra.resolve_workspace_args(args)
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
        workspace_root=workspace_path,
    ).enforce(apply_changes=apply_changes)
    output.metrics(
        {"workspace": workspace_path, "mode": "apply" if apply_changes else "dry-run"},
        centralize_summary,
        {
            "mro_remaining_violations": mro_report.remaining_violations,
            "mro_files_scanned": mro_report.files_scanned,
            "mro_files_with_candidates": mro_report.files_with_candidates,
            "mro_failures": mro_report.mro_failures,
        },
        {
            "namespace_loose_objects": namespace_report.total_loose_objects,
            "namespace_import_violations": namespace_report.total_import_violations,
            "namespace_cyclic_imports": namespace_report.total_cyclic_imports,
            "namespace_runtime_alias_violations": namespace_report.total_runtime_alias_violations,
            "namespace_manual_protocols": namespace_report.total_manual_protocol_violations,
            "namespace_manual_typing_aliases": namespace_report.total_manual_typing_violations,
            "namespace_compatibility_aliases": namespace_report.total_compatibility_alias_violations,
            "namespace_parse_failures": namespace_report.total_parse_failures,
            "namespace_files_scanned": namespace_report.total_files_scanned,
        },
    )
    if len(mro_report.errors) > 0:
        for error in mro_report.errors:
            output.error(error)
        return 1
    return 0


def _run_census(*, argv: list[str]) -> int:
    parser = u.Infra.create_refactor_parser(
        prog="flext_infra refactor census",
        description="Run AST/CST census of MRO family method usage across workspace projects",
        include_apply=False,
    )
    _ = parser.add_argument(
        "--family",
        type=str,
        default="u",
        choices=sorted({"c", "t", "p", "m", "u"}),
        help="MRO family to census (default: u). Options: c, t, p, m, u",
    )
    _ = parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Path to write JSON report (optional)",
    )
    args = parser.parse_args(argv)
    workspace_path = args.workspace.resolve()
    census = FlextInfraRefactorCensus()

    target = u.Infra.build_mro_target(args.family)
    result = census.run(workspace_path, target=target)
    if result.is_failure:
        output.error(f"Census failed: {result.error}")
        return 1
    report = result.value
    output.write(FlextInfraRefactorCensus.render_text(report))
    output.write("\n")
    if args.json_output:
        json_path = Path(args.json_output).resolve()
        u.Infra.export_pydantic_json(report, json_path)
        output.info(f"JSON report exported to: {json_path}")

    output.metrics(
        {"family": args.family, "workspace": workspace_path},
        report,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
