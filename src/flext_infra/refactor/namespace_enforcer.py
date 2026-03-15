"""Automated namespace enforcement orchestration."""

from __future__ import annotations

from pathlib import Path

from flext_infra import c, u
from flext_infra.refactor._models_namespace_enforcer import (
    FlextInfraNamespaceEnforcerModels as nem,
)
from flext_infra.refactor.dependency_analyzer import (
    FlextInfraRefactorDependencyAnalyzerFacade,
)
from flext_infra.refactor.namespace_rewriter import NamespaceEnforcementRewriter
from flext_infra.refactor.output import FlextInfraRefactorOutputRenderer


class FlextInfraNamespaceEnforcer:
    """Orchestrate namespace enforcement across a workspace."""

    def __init__(self, *, workspace_root: Path) -> None:
        """Initialize with the workspace root path."""
        super().__init__()
        self._workspace_root = workspace_root.resolve()

    def enforce(
        self,
        *,
        apply: bool = False,
    ) -> nem.WorkspaceEnforcementReport:
        """Run namespace enforcement across all projects in the workspace."""
        project_roots = u.Infra.discover_project_roots(
            workspace_root=self._workspace_root,
        )
        project_reports: list[nem.ProjectEnforcementReport] = []
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
                apply=apply,
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
        return nem.WorkspaceEnforcementReport.create(
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
        apply: bool,
    ) -> nem.ProjectEnforcementReport:
        parse_failures: list[nem.ParseFailureViolation] = []
        facade_statuses = FlextInfraRefactorDependencyAnalyzerFacade.NamespaceFacadeScanner.scan_project(
            project_root=project_root,
            project_name=project_name,
            parse_failures=parse_failures,
        )
        if apply:
            NamespaceEnforcementRewriter.ensure_missing_facades(
                project_root=project_root,
                project_name=project_name,
                facade_statuses=facade_statuses,
            )
            facade_statuses = FlextInfraRefactorDependencyAnalyzerFacade.NamespaceFacadeScanner.scan_project(
                project_root=project_root,
                project_name=project_name,
                parse_failures=parse_failures,
            )
        py_files_result = u.Infra.iter_python_files(
            workspace_root=project_root,
            project_roots=[project_root],
            src_dirs=frozenset(c.Infra.Refactor.MRO_SCAN_DIRECTORIES),
        )
        py_files = [] if py_files_result.is_failure else py_files_result.value
        loose_objects: list[nem.LooseObjectViolation] = []
        for py_file in py_files:
            loose_objects.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.LooseObjectDetector.detect_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                ),
            )
        import_violations: list[nem.ImportAliasViolation] = []
        for py_file in py_files:
            import_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.ImportAliasDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(import_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_import_alias_violations(
                py_files=py_files,
            )
            import_violations = []
            for py_file in py_files:
                import_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.ImportAliasDetector.detect_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    ),
                )
        cyclic_imports = FlextInfraRefactorDependencyAnalyzerFacade.CyclicImportDetector.scan_project(
            project_root=project_root,
            _parse_failures=parse_failures,
        )
        internal_import_violations: list[nem.InternalImportViolation] = []
        for py_file in py_files:
            internal_import_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.InternalImportDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        runtime_alias_violations: list[nem.RuntimeAliasViolation] = []
        for py_file in py_files:
            runtime_alias_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.RuntimeAliasDetector.detect_file(
                    file_path=py_file,
                    project_name=project_name,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(runtime_alias_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_runtime_alias_violations(
                py_files=py_files,
            )
            runtime_alias_violations = []
            for py_file in py_files:
                runtime_alias_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.RuntimeAliasDetector.detect_file(
                        file_path=py_file,
                        project_name=project_name,
                        parse_failures=parse_failures,
                    ),
                )
        future_violations: list[nem.FutureAnnotationsViolation] = []
        for py_file in py_files:
            future_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.FutureAnnotationsDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(future_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_missing_future_annotations(
                py_files=py_files,
            )
            future_violations = []
            for py_file in py_files:
                future_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.FutureAnnotationsDetector.detect_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    ),
                )
        manual_protocol_violations: list[nem.ManualProtocolViolation] = []
        for py_file in py_files:
            manual_protocol_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.ManualProtocolDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(manual_protocol_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_manual_protocol_violations(
                project_root=project_root,
                py_files=py_files,
                violations=manual_protocol_violations,
            )
            manual_protocol_violations = []
            for py_file in py_files:
                manual_protocol_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.ManualProtocolDetector.detect_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    ),
                )
        manual_typing_violations: list[nem.ManualTypingAliasViolation] = []
        for py_file in py_files:
            manual_typing_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.ManualTypingAliasDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(manual_typing_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_manual_typing_alias_violations(
                project_root=project_root,
                violations=manual_typing_violations,
                parse_failures=parse_failures,
            )
            manual_typing_violations = []
            for py_file in py_files:
                manual_typing_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.ManualTypingAliasDetector.detect_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    ),
                )
        compatibility_alias_violations: list[nem.CompatibilityAliasViolation] = []
        for py_file in py_files:
            compatibility_alias_violations.extend(
                FlextInfraRefactorDependencyAnalyzerFacade.CompatibilityAliasDetector.detect_file(
                    file_path=py_file,
                    parse_failures=parse_failures,
                ),
            )
        if apply and len(compatibility_alias_violations) > 0:
            NamespaceEnforcementRewriter.rewrite_compatibility_alias_violations(
                violations=compatibility_alias_violations,
                parse_failures=parse_failures,
            )
            compatibility_alias_violations = []
            for py_file in py_files:
                compatibility_alias_violations.extend(
                    FlextInfraRefactorDependencyAnalyzerFacade.CompatibilityAliasDetector.detect_file(
                        file_path=py_file,
                        parse_failures=parse_failures,
                    ),
                )
        return nem.ProjectEnforcementReport.create(
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
        report: nem.WorkspaceEnforcementReport,
    ) -> str:
        """Render a workspace enforcement report as plain text."""
        return FlextInfraRefactorOutputRenderer.render_namespace_enforcement_report(
            report
        )


__all__ = ["FlextInfraNamespaceEnforcer"]
