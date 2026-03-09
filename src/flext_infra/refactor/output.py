"""FLEXT infrastructure refactor output rendering."""

from __future__ import annotations

from flext_infra import c, m


def render_namespace_enforcement_report(
    report: m.Infra.Refactor.NamespaceEnforcementModels.WorkspaceEnforcementReport,
) -> str:
    """Render a human-readable namespace enforcement report."""
    max_loose = c.Infra.Refactor.NAMESPACE_MAX_RENDERED_LOOSE_OBJECTS
    max_imports = c.Infra.Refactor.NAMESPACE_MAX_RENDERED_IMPORT_VIOLATIONS
    lines: list[str] = [
        f"Workspace: {report.workspace}",
        f"Projects scanned: {len(report.projects)}",
        f"Files scanned: {report.total_files_scanned}",
        f"Missing facades: {report.total_facades_missing}",
        f"Loose objects: {report.total_loose_objects}",
        f"Import violations: {report.total_import_violations}",
        f"Internal imports: {report.total_internal_import_violations}",
        f"Cyclic imports: {report.total_cyclic_imports}",
        f"Runtime alias violations: {report.total_runtime_alias_violations}",
        f"Missing __future__: {report.total_future_violations}",
        f"Manual protocols: {report.total_manual_protocol_violations}",
        f"Manual typing aliases: {report.total_manual_typing_violations}",
        f"Compatibility aliases: {report.total_compatibility_alias_violations}",
        f"Parse failures: {report.total_parse_failures}",
        "",
    ]
    for proj in report.projects:
        missing = [s for s in proj.facade_statuses if not s.exists]
        has_violations = (
            missing
            or proj.loose_objects
            or proj.import_violations
            or proj.internal_import_violations
            or proj.runtime_alias_violations
            or proj.future_violations
            or proj.manual_protocol_violations
            or proj.manual_typing_violations
            or proj.compatibility_alias_violations
            or proj.parse_failures
        )
        if not has_violations:
            continue
        lines.append(f"--- {proj.project} ---")
        if missing:
            lines.append(
                "  Missing facades: "
                + ", ".join(
                    f"{s.family} ({c.Infra.Refactor.NAMESPACE_FACADE_FAMILIES[s.family]})"
                    for s in missing
                ),
            )
        if proj.loose_objects:
            lines.append(f"  Loose objects: {len(proj.loose_objects)}")
            lines.extend(
                f"    {obj.file}:{obj.line} {obj.kind} '{obj.name}' -> {obj.suggestion}"
                for obj in proj.loose_objects[:max_loose]
            )
            if len(proj.loose_objects) > max_loose:
                lines.append(f"    ... and {len(proj.loose_objects) - max_loose} more")
        if proj.import_violations:
            lines.append(f"  Import violations: {len(proj.import_violations)}")
            lines.extend(
                f"    {iv.file}:{iv.line} {iv.current_import}"
                for iv in proj.import_violations[:max_imports]
            )
            if len(proj.import_violations) > max_imports:
                lines.append(
                    f"    ... and {len(proj.import_violations) - max_imports} more",
                )
        if proj.internal_import_violations:
            lines.append(f"  Internal imports: {len(proj.internal_import_violations)}")
            lines.extend(
                f"    {iv.file}:{iv.line} {iv.current_import} ({iv.detail})"
                for iv in proj.internal_import_violations[:max_imports]
            )
            if len(proj.internal_import_violations) > max_imports:
                lines.append(
                    f"    ... and {len(proj.internal_import_violations) - max_imports} more",
                )
        if proj.cyclic_imports:
            lines.append(f"  Cyclic imports: {len(proj.cyclic_imports)}")
            lines.extend(
                f"    Cycle: {' -> '.join(ci.cycle)}" for ci in proj.cyclic_imports
            )
        if proj.runtime_alias_violations:
            lines.append(
                f"  Runtime alias violations: {len(proj.runtime_alias_violations)}",
            )
            lines.extend(
                f"    {rv.file} [{rv.kind}] alias='{rv.alias}' {rv.detail}"
                for rv in proj.runtime_alias_violations
            )
        if proj.future_violations:
            lines.append(
                f"  Missing __future__ annotations: {len(proj.future_violations)}",
            )
            lines.extend(f"    {fv.file}" for fv in proj.future_violations[:max_loose])
            if len(proj.future_violations) > max_loose:
                lines.append(
                    f"    ... and {len(proj.future_violations) - max_loose} more",
                )
        if proj.manual_protocol_violations:
            lines.append(f"  Manual protocols: {len(proj.manual_protocol_violations)}")
            lines.extend(
                f"    {pv.file}:{pv.line} {pv.name}"
                for pv in proj.manual_protocol_violations[:max_loose]
            )
            if len(proj.manual_protocol_violations) > max_loose:
                lines.append(
                    f"    ... and {len(proj.manual_protocol_violations) - max_loose} more",
                )
        if proj.manual_typing_violations:
            lines.append(
                f"  Manual typing aliases: {len(proj.manual_typing_violations)}",
            )
            lines.extend(
                f"    {tv.file}:{tv.line} {tv.name}"
                for tv in proj.manual_typing_violations[:max_loose]
            )
            if len(proj.manual_typing_violations) > max_loose:
                lines.append(
                    f"    ... and {len(proj.manual_typing_violations) - max_loose} more",
                )
        if proj.compatibility_alias_violations:
            lines.append(
                f"  Compatibility aliases: {len(proj.compatibility_alias_violations)}",
            )
            lines.extend(
                f"    {cv.file}:{cv.line} {cv.alias_name}={cv.target_name}"
                for cv in proj.compatibility_alias_violations[:max_loose]
            )
            if len(proj.compatibility_alias_violations) > max_loose:
                lines.append(
                    f"    ... and {len(proj.compatibility_alias_violations) - max_loose} more",
                )
        if proj.parse_failures:
            lines.append(f"  Parse failures: {len(proj.parse_failures)}")
            lines.extend(
                f"    {pf.file} [{pf.stage}] {pf.error_type}: {pf.detail}"
                for pf in proj.parse_failures[:max_loose]
            )
            if len(proj.parse_failures) > max_loose:
                lines.append(f"    ... and {len(proj.parse_failures) - max_loose} more")
        lines.append("")
    return "\n".join(lines) + "\n"


__all__ = ["render_namespace_enforcement_report"]
