"""Source-specific enforcement violation collectors."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from flext_infra import c, m, u
from flext_infra._enforcement.collection_base import (
    FlextInfraEnforcementCollectionBase,
    FlextInfraEnforcementEvaluation,
)
from flext_infra._enforcement.collection_tests import (
    FlextInfraEnforcementTestsCollector,
)
from flext_infra._enforcement.metadata import FlextInfraEnforcementMetadata
from flext_infra._enforcement.selection import FlextInfraEnforcementSelection

if TYPE_CHECKING:
    from flext_core._models.enforcement import FlextModelsEnforcement as me
    from flext_infra import p, t


class FlextInfraEnforcementSourceCollectors(
    FlextInfraEnforcementMetadata,
    FlextInfraEnforcementSelection,
    FlextInfraEnforcementTestsCollector,
    FlextInfraEnforcementCollectionBase,
):
    """Collect enforcement probes for supported catalog source kinds."""

    def __init__(self, repository_root: Path) -> None:
        """Bind the repository root used for project discovery and Rope sessions."""
        self._repository_root = repository_root

    def collect_project(
        self, project_dir: Path, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> FlextInfraEnforcementEvaluation:
        """Collect rule probes for one project using one shared dispatcher."""
        violations: list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]] = []
        failures: list[m.Infra.FailedFix] = []
        declarative_rules: list[me.EnforcementRuleSpec] = []
        for rule in rules:
            source = rule.source
            if source.kind == "flext_tests_validator":
                collected, errors = self.collect_tests_validator(project_dir, rule)
            elif self.supports_declarative(rule):
                declarative_rules.append(rule)
                continue
            elif source.kind in {"flext_infra_detector", "beartype"}:
                collected, errors = self.collect_python_file_probes(project_dir, rule)
            elif source.kind in {"ruff", "code_smell"}:
                violations.extend(self.collect_project_probe(project_dir, rule))
                continue
            else:
                failures.append(
                    self.collection_failure(
                        project_dir,
                        rule,
                        f"unsupported enforcement source kind {source.kind!r}",
                    )
                )
                continue
            violations.extend(collected)
            failures.extend(errors)
        if declarative_rules:
            collected, errors = self.collect_declarative(project_dir, declarative_rules)
            violations.extend(collected)
            failures.extend(errors)
        return FlextInfraEnforcementEvaluation(violations, failures)

    def collect_python_file_probes(
        self, project_dir: Path, rule: me.EnforcementRuleSpec
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Return one structural probe per Python file for file-wide transformers."""
        files_result = u.Infra.iter_python_files(
            m.Infra.SourceScanRequest(project_roots=(project_dir,))
        )
        if files_result.failure:
            return self._empty_failure(
                project_dir,
                rule,
                files_result.error or "unable to enumerate Python files",
            )
        return ([(rule, self.probe_for_path(path)) for path in files_result.value], [])

    def collect_declarative(
        self, project_dir: Path, rules: t.SequenceOf[me.EnforcementRuleSpec]
    ) -> tuple[
        list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]], list[m.Infra.FailedFix]
    ]:
        """Run catalog-driven declarative rules across one project."""
        files, errors = self.collect_python_file_probes(project_dir, rules[0])
        if errors:
            return [], errors
        file_paths = [
            Path(str(getattr(probe, "file_path", ""))) for _rule, probe in files
        ]
        if any(self.rule_requires_stub_file(rule) for rule in rules):
            file_paths.extend(self.stub_file_paths(project_dir))
        probes: list[tuple[me.EnforcementRuleSpec, p.AttributeProbe]] = []
        failures: list[m.Infra.FailedFix] = []
        with u.Infra.open_project(self._repository_root) as rope_project:
            for file_path in file_paths:
                ctx = m.Infra.DetectorContext(
                    file_path=file_path,
                    rope_project=rope_project,
                    project_name=project_dir.name,
                    project_root=project_dir,
                )
                for rule in rules:
                    try:
                        detected = self.detect_declarative(rule, ctx)
                    except c.EXC_BROAD_RUNTIME as exc:
                        failures.append(
                            self.collection_failure(
                                project_dir, rule, f"declarative engine failed: {exc}"
                            )
                        )
                        continue
                    probes.extend((rule, probe) for probe in detected)
        return probes, failures


__all__: list[str] = ["FlextInfraEnforcementSourceCollectors"]
