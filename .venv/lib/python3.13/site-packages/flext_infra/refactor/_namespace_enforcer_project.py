"""Per-project namespace enforcement — extracted concern of the namespace enforcer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flext_infra import config, m, u
from flext_infra.detectors.class_placement_detector import (
    FlextInfraClassPlacementDetector,
)
from flext_infra.detectors.compatibility_alias_detector import (
    FlextInfraCompatibilityAliasDetector,
)
from flext_infra.detectors.cyclic_import_detector import FlextInfraCyclicImportDetector
from flext_infra.detectors.future_annotations_detector import (
    FlextInfraFutureAnnotationsDetector,
)
from flext_infra.detectors.import_alias_detector import FlextInfraImportAliasDetector
from flext_infra.detectors.internal_import_detector import (
    FlextInfraInternalImportDetector,
)
from flext_infra.detectors.loose_object_detector import FlextInfraLooseObjectDetector
from flext_infra.detectors.manual_protocol_detector import (
    FlextInfraManualProtocolDetector,
)
from flext_infra.detectors.manual_typing_alias_detector import (
    FlextInfraManualTypingAliasDetector,
)
from flext_infra.detectors.namespace_source_detector import (
    FlextInfraNamespaceSourceDetector,
)
from flext_infra.detectors.private_import_bypass_detector import (
    FlextInfraPrivateImportBypassDetector,
)
from flext_infra.detectors.runtime_alias_detector import FlextInfraRuntimeAliasDetector

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from flext_infra import t


class FlextInfraNamespaceEnforcerProjectMixin:
    """Run every detector over one project and build its enforcement report.

    Composed alongside the phases/orchestration mixins into the concrete
    enforcer; ``self`` provides facade scan / file-collection / detect-apply
    helpers through the concrete's FLEXT.
    """

    if TYPE_CHECKING:
        _repository_root: Path
        _rope_project: t.Infra.RopeProject

        def _detect_and_apply[V](
            self,
            *,
            py_files: t.SequenceOf[Path],
            detect_fn: Callable[[Path], t.SequenceOf[V]],
            rewrite_fn: Callable[[t.MutableSequenceOf[V]], None] | None,
            apply: bool,
        ) -> t.MutableSequenceOf[V]: ...

        @staticmethod
        def _scan_facades(
            *,
            project: tuple[Path, str],
            rope_project: t.Infra.RopeProject,
            apply: bool,
            repository_root: Path,
        ) -> t.SequenceOf[m.Infra.FacadeStatus]: ...

        @staticmethod
        def _collect_py_files(*, project_root: Path) -> t.SequenceOf[Path]: ...

    def _enforce_project(
        self,
        *,
        project_root: Path,
        project_name: str,
        apply: bool,
        gates: t.StrSequence | None = None,
    ) -> m.Infra.ProjectEnforcementReport:
        """Enforce project."""
        return self._enforce_project_with_rope(
            project_root=project_root,
            project_name=project_name,
            apply=apply,
            gates=gates,
            rope_project=self._rope_project,
        )

    @staticmethod
    def _detector_context(
        *,
        file_path: Path,
        rope_project: t.Infra.RopeProject,
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation],
        project_name: str = "",
        project_root: Path | None = None,
    ) -> m.Infra.DetectorContext:
        """Build the canonical detector context for one file."""
        return m.Infra.DetectorContext(
            file_path=file_path,
            rope_project=rope_project,
            parse_failures=parse_failures,
            project_name=project_name,
            project_root=project_root,
        )

    def _enforce_project_with_rope(
        self,
        *,
        project_root: Path,
        project_name: str,
        apply: bool,
        gates: t.StrSequence | None,
        rope_project: t.Infra.RopeProject,
    ) -> m.Infra.ProjectEnforcementReport:
        """Enforce project using the Rope project scoped to ``project_root``."""
        parse_failures: t.MutableSequenceOf[m.Infra.ParseFailureViolation] = []
        facade_statuses = self._scan_facades(
            project=(project_root, project_name),
            rope_project=rope_project,
            apply=apply,
            repository_root=self._repository_root,
        )
        py_files = self._collect_py_files(project_root=project_root)
        project_layout = u.Infra.layout(project_root)
        package_name = project_layout.package_name if project_layout is not None else ""

        loose_objects = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraLooseObjectDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_name=project_name,
                    project_root=project_root,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_loose_object_violations(
                project_root=project_root,
                violations=vs,
                parse_failures=parse_failures,
                gates=gates,
            ),
            apply=apply,
        )
        import_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraImportAliasDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                )
            ),
            rewrite_fn=lambda _vs: u.Infra.rewrite_import_violations(
                py_files=py_files, project_package=package_name
            ),
            apply=apply,
        )
        namespace_source_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraNamespaceSourceDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_name=project_name,
                    project_root=project_root,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_namespace_source_violations(
                violations=vs, parse_failures=parse_failures, gates=gates
            ),
            apply=apply,
        )
        cyclic_imports = FlextInfraCyclicImportDetector.scan_project(
            project_root=project_root,
            rope_project=rope_project,
            _parse_failures=parse_failures,
        )
        internal_import_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraInternalImportDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_root=project_root,
                )
            ),
            rewrite_fn=None,
            apply=apply,
        )
        private_import_bypass_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraPrivateImportBypassDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_root=project_root,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_private_import_bypass_violations(
                rope_project=rope_project,
                violations=tuple(v for v in vs if v.symbol_exported),
                parse_failures=parse_failures,
            ),
            apply=apply,
        )
        runtime_alias_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraRuntimeAliasDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_name=project_name,
                    project_root=project_root,
                )
            ),
            rewrite_fn=lambda _vs: u.Infra.rewrite_runtime_alias_violations(
                py_files=py_files, gates=gates
            ),
            apply=apply,
        )
        future_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraFutureAnnotationsDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_root=project_root,
                )
            ),
            rewrite_fn=lambda _vs: u.Infra.rewrite_missing_future_annotations(
                py_files=py_files
            ),
            apply=apply,
        )
        manual_protocol_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraManualProtocolDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_manual_protocol_violations(
                project_root=project_root, py_files=py_files, violations=vs, gates=gates
            ),
            apply=apply,
        )
        manual_typing_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraManualTypingAliasDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_manual_typing_alias_violations(
                project_root=project_root,
                violations=vs,
                parse_failures=parse_failures,
                gates=gates,
            ),
            apply=apply,
        )
        compatibility_alias_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraCompatibilityAliasDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                )
            ),
            rewrite_fn=lambda vs: u.Infra.rewrite_compatibility_alias_violations(
                violations=vs, parse_failures=parse_failures, gates=gates
            ),
            apply=apply,
        )
        (compatibility_alias_violations, foreign_canonical_alias_violations) = (
            self._split_compatibility_alias_violations(
                compatibility_alias_violations, package_name=package_name
            )
        )
        class_placement_violations = self._detect_and_apply(
            py_files=py_files,
            detect_fn=lambda f: FlextInfraClassPlacementDetector.detect_file(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                )
            ),
            rewrite_fn=None,
            apply=apply,
        )
        pattern_smells = self._detect_and_apply(
            py_files=py_files,
            # flext-j47u (codex): config data + u.Infra are the only static-policy path.
            detect_fn=lambda f: u.Infra.detect_static_rules(
                self._detector_context(
                    file_path=f,
                    rope_project=rope_project,
                    parse_failures=parse_failures,
                    project_name=project_name,
                    project_root=project_root,
                ),
                config.Infra.enforcement.rules,
            ),
            rewrite_fn=None,
            apply=apply,
        )
        smell_buckets: dict[str, list[m.Infra.PatternSmellViolation]] = {
            "bare_except": [],
            "print": [],
            "breakpoint": [],
            "open_encoding": [],
            "dict_annotation": [],
            "typing_dict_attr": [],
            "typing_dict_import": [],
            "hardcoded_version": [],
            "type_ignore": [],
            "noqa": [],
        }
        for smell in pattern_smells:
            bucket = smell_buckets.get(smell.kind)
            if bucket is not None:
                bucket.append(smell)
        return m.Infra.ProjectEnforcementReport(
            project=project_name,
            project_root=str(project_root),
            facade_statuses=facade_statuses,
            loose_objects=list(loose_objects),
            import_violations=list(import_violations),
            namespace_source_violations=list(namespace_source_violations),
            internal_import_violations=list(internal_import_violations),
            private_import_bypass_violations=list(private_import_bypass_violations),
            manual_protocol_violations=list(manual_protocol_violations),
            cyclic_imports=list(cyclic_imports),
            runtime_alias_violations=list(runtime_alias_violations),
            future_violations=list(future_violations),
            manual_typing_violations=list(manual_typing_violations),
            compatibility_alias_violations=list(compatibility_alias_violations),
            foreign_canonical_alias_violations=list(foreign_canonical_alias_violations),
            class_placement_violations=list(class_placement_violations),
            bare_except_violations=smell_buckets["bare_except"],
            print_violations=smell_buckets["print"],
            breakpoint_violations=smell_buckets["breakpoint"],
            open_encoding_violations=smell_buckets["open_encoding"],
            dict_annotation_violations=smell_buckets["dict_annotation"],
            typing_dict_attr_violations=smell_buckets["typing_dict_attr"],
            typing_dict_import_violations=smell_buckets["typing_dict_import"],
            hardcoded_version_violations=smell_buckets["hardcoded_version"],
            type_ignore_violations=smell_buckets["type_ignore"],
            noqa_violations=smell_buckets["noqa"],
            parse_failures=list(parse_failures),
            files_scanned=len(py_files),
        )

    @staticmethod
    def _split_compatibility_alias_violations(
        violations: t.SequenceOf[m.Infra.CompatibilityAliasViolation],
        *,
        package_name: str,
    ) -> tuple[
        list[m.Infra.CompatibilityAliasViolation],
        list[m.Infra.CompatibilityAliasViolation],
    ]:
        """Split legacy compatibility aliases from foreign canonical imports."""
        compatibility_aliases: list[m.Infra.CompatibilityAliasViolation] = []
        foreign_canonical_aliases: list[m.Infra.CompatibilityAliasViolation] = []
        for violation in violations:
            action = FlextInfraCompatibilityAliasDetector.fix_action_for(
                violation, current_project=package_name
            )
            if action == "rewrite_foreign_canonical_alias":
                foreign_canonical_aliases.append(violation)
                continue
            compatibility_aliases.append(violation)
        return compatibility_aliases, foreign_canonical_aliases


__all__: list[str] = ["FlextInfraNamespaceEnforcerProjectMixin"]
