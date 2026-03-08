"""Automated namespace enforcement orchestration."""

from __future__ import annotations

from pathlib import Path

from flext_infra import m, u
from flext_infra.refactor.analysis import (
    CompatibilityAliasDetector,
    CyclicImportDetector,
    FutureAnnotationsDetector,
    ImportAliasDetector,
    InternalImportDetector,
    LooseObjectDetector,
    ManualProtocolDetector,
    ManualTypingAliasDetector,
    NamespaceFacadeScanner,
    RuntimeAliasDetector,
)
from flext_infra.refactor.namespace_rewriter import NamespaceEnforcementRewriter
from flext_infra.refactor.output import render_namespace_enforcement_report


NamespaceEnforcementModels = m.Infra.Refactor.NamespaceEnforcementModels


class FlextInfraNamespaceEnforcer:
    def __init__(self, *, workspace_root: Path) -> None:
        super().__init__()
        self._workspace_root = workspace_root.resolve()

    def enforce(
        self, *, apply_changes: bool = False
    ) -> NamespaceEnforcementModels.WorkspaceEnforcementReport:
        project_roots = u.Infra.Refactor.discover_project_roots(
            workspace_root=self._workspace_root
        )
        project_reports: list[NamespaceEnforcementModels.ProjectEnforcementReport] = []
        total_missing = 0
        total_loose = 0
        total_import_v = 0
        total_internal_import_v = 0
        total_cyclic = 0
        total_alias_v = 0
        total_future_v = 0
        total_manual_protocol_v = 0
        total_manual_typing_v = 0
        total_compat_alias_v = 0
        total_parse_failures = 0
        total_files = 0
        for project_root in project_roots:
            project_name = project_root.name
            report = self._enforce_project(
                project_root=project_root,
                project_name=project_name,
                apply_changes=apply_changes,
            )
            project_reports.append(report)
            total_missing += sum(1 for s in report.facade_statuses if not s.exists)
            total_loose += len(report.loose_objects)
            total_import_v += len(report.import_violations)
            total_internal_import_v += len(report.internal_import_violations)
            total_cyclic += len(report.cyclic_imports)
            total_alias_v += len(report.runtime_alias_violations)
            total_future_v += len(report.future_violations)
            total_manual_protocol_v += len(report.manual_protocol_violations)
            total_manual_typing_v += len(report.manual_typing_violations)
            total_compat_alias_v += len(report.compatibility_alias_violations)
            total_parse_failures += len(report.parse_failures)
            total_files += report.files_scanned
        return NamespaceEnforcementModels.WorkspaceEnforcementReport.create(
            workspace=str(self._workspace_root),
            projects=project_reports,
            total_facades_missing=total_missing,
            total_loose_objects=total_loose,
            total_import_violations=total_import_v,
            total_internal_import_violations=total_internal_import_v,
            total_cyclic_imports=total_cyclic,
            total_runtime_alias_violations=total_alias_v,
            total_future_violations=total_future_v,
            total_manual_protocol_violations=total_manual_protocol_v,
            total_manual_typing_violations=total_manual_typing_v,
            total_compatibility_alias_violations=total_compat_alias_v,
            total_parse_failures=total_parse_failures,
            total_files_scanned=total_files,
        )

    def _enforce_project(
        self,
        *,
        project_root: Path,
        project_name: str,
        apply_changes: bool,
    ) -> NamespaceEnforcementModels.ProjectEnforcementReport:
        parse_failures: list[NamespaceEnforcementModels.ParseFailureViolation] = []
        facade_statuses = NamespaceFacadeScanner.scan_project(
            project_root=project_root,
            project_name=project_name,
            parse_failures=parse_failures,
        )
        if apply_changes:
            NamespaceEnforcementRewriter.ensure_missing_facades(
                project_root=project_root,
                project_name=project_name,
                facade_statuses=facade_statuses,
            )
            facade_statuses = NamespaceFacadeScanner.scan_project(
                project_root=project_root,
                project_name=project_name,
                parse_failures=parse_failures,
            )
        py_files = NamespaceEnforcementRewriter.collect_python_files(
            project_root=project_root
        )
        loose_objects: list[NamespaceEnforcementModels.LooseObjectViolation] = []
        for py_file in py_files:
            loose_objects.extend(
                LooseObjectDetector.scan_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                )
            )
        import_violations: list[NamespaceEnforcementModels.ImportAliasViolation] = []
        for py_file in py_files:
            import_violations.extend(
                ImportAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(import_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_import_alias_violations(
                py_files=py_files
            )
            import_violations = []
            for py_file in py_files:
                import_violations.extend(
                    ImportAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )
        cyclic_imports = CyclicImportDetector.scan_project(
            project_root=project_root,
            parse_failures=parse_failures,
        )
        internal_import_violations: list[
            NamespaceEnforcementModels.InternalImportViolation
        ] = []
        for py_file in py_files:
            internal_import_violations.extend(
                InternalImportDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        runtime_alias_violations: list[
            NamespaceEnforcementModels.RuntimeAliasViolation
        ] = []
        for py_file in py_files:
            runtime_alias_violations.extend(
                RuntimeAliasDetector.scan_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(runtime_alias_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_runtime_alias_violations(
                py_files=py_files
            )
            runtime_alias_violations = []
            for py_file in py_files:
                runtime_alias_violations.extend(
                    RuntimeAliasDetector.scan_file(
                        file_path=py_file,
                        project_name=project_name,
                        parse_failures=parse_failures,
                    )
                )
        future_violations: list[
            NamespaceEnforcementModels.FutureAnnotationsViolation
        ] = []
        for py_file in py_files:
            future_violations.extend(
                FutureAnnotationsDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(future_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_missing_future_annotations(
                py_files=py_files
            )
            future_violations = []
            for py_file in py_files:
                future_violations.extend(
                    FutureAnnotationsDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )
        manual_protocol_violations: list[
            NamespaceEnforcementModels.ManualProtocolViolation
        ] = []
        for py_file in py_files:
            manual_protocol_violations.extend(
                ManualProtocolDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(manual_protocol_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_manual_protocol_violations(
                project_root=project_root,
                py_files=py_files,
                violations=manual_protocol_violations,
            )
            manual_protocol_violations = []
            for py_file in py_files:
                manual_protocol_violations.extend(
                    ManualProtocolDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )
        manual_typing_violations: list[
            NamespaceEnforcementModels.ManualTypingAliasViolation
        ] = []
        for py_file in py_files:
            manual_typing_violations.extend(
                ManualTypingAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(manual_typing_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_manual_typing_alias_violations(
                project_root=project_root,
                violations=manual_typing_violations,
                parse_failures=parse_failures,
            )
            manual_typing_violations = []
            for py_file in py_files:
                manual_typing_violations.extend(
                    ManualTypingAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )
        compatibility_alias_violations: list[
            NamespaceEnforcementModels.CompatibilityAliasViolation
        ] = []
        for py_file in py_files:
            compatibility_alias_violations.extend(
                CompatibilityAliasDetector.scan_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                )
            )
        if apply_changes and len(compatibility_alias_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_compatibility_alias_violations(
                violations=compatibility_alias_violations,
                parse_failures=parse_failures,
            )
            compatibility_alias_violations = []
            for py_file in py_files:
                compatibility_alias_violations.extend(
                    CompatibilityAliasDetector.scan_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    )
                )
        return NamespaceEnforcementModels.ProjectEnforcementReport.create(
            project=project_name,
            project_root=str(project_root),
            facade_statuses=facade_statuses,
            loose_objects=loose_objects,
            import_violations=import_violations,
            internal_import_violations=internal_import_violations,
            manual_protocol_violations=manual_protocol_violations,
            cyclic_imports=cyclic_imports,
            runtime_alias_violations=runtime_alias_violations,
            future_violations=future_violations,
            manual_typing_violations=manual_typing_violations,
            compatibility_alias_violations=compatibility_alias_violations,
            parse_failures=parse_failures,
            files_scanned=len(py_files),
        )

    @staticmethod
    def render_text(
        report: NamespaceEnforcementModels.WorkspaceEnforcementReport,
    ) -> str:
        return render_namespace_enforcement_report(report)


__all__ = [
    "CompatibilityAliasDetector",
    "CyclicImportDetector",
    "FlextInfraNamespaceEnforcer",
    "FutureAnnotationsDetector",
    "ImportAliasDetector",
    "InternalImportDetector",
    "LooseObjectDetector",
    "ManualProtocolDetector",
    "ManualTypingAliasDetector",
    "NamespaceEnforcementModels",
    "NamespaceFacadeScanner",
    "RuntimeAliasDetector",
]
