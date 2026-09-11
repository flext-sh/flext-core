"""Rope-semantic guard for the strict package-test import DAG."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, override

from flext_core import r
from flext_infra import m, t, u
from flext_infra.base import s

if TYPE_CHECKING:
    from flext_infra import p


class FlextInfraValidateTestImportDag(s[bool]):
    """Enforce directed imports between production, tests, and test facets."""

    _RULES_FILE: ClassVar[Path] = (
        Path(__file__).parent.parent / "rules" / "test-import-dag.yml"
    )
    _rules_cache: ClassVar[m.Infra.TestImportDagRulesConfig | None] = None

    @classmethod
    def _rules(cls) -> m.Infra.TestImportDagRulesConfig:
        cached = cls._rules_cache
        if cached is None:
            cached = m.Infra.TestImportDagRulesConfig.model_validate(
                u.Cli.yaml_load_mapping(cls._RULES_FILE)
            )
            cls._rules_cache = cached
        return cached

    def build_report(self, repository_root: Path) -> p.Result[m.Infra.ValidationReport]:
        """Scan every governed project as an independent import unit."""
        try:
            roots = u.Infra.discover_project_roots(repository_root) or (
                repository_root,
            )
            violations = tuple(
                violation
                for project_root in roots
                for violation in self._project_violations(project_root)
            )
        except OSError as exc:
            return r[m.Infra.ValidationReport].fail_op("test-import-dag scan", exc)
        summary = (
            f"strict test import DAG respected ({len(roots)} project(s))"
            if not violations
            else f"{len(violations)} strict test import DAG violation(s)"
        )
        return r[m.Infra.ValidationReport].ok(
            m.Infra.ValidationReport(
                passed=not violations, violations=violations, summary=summary
            )
        )

    def _project_violations(self, project_root: Path) -> t.StrSequence:
        rules = self._rules()
        violations: list[str] = []
        with u.Infra.open_project(project_root) as project:
            for resource in u.Infra.python_resources(project):
                file_path = u.Infra.resource_file_path(project, resource)
                if file_path is None:
                    continue
                module_imports = u.Infra.get_module_imports(project, resource)
                if module_imports is None:
                    continue
                for imported in u.Infra.imported_module_paths(module_imports):
                    reason = self._edge_violation(
                        file_path, imported, project_root, rules
                    )
                    if reason is not None:
                        violations.append(f"{file_path}: {imported}: {reason}")
        return tuple(violations)

    @staticmethod
    def _facet(file_path: Path, rules: m.Infra.TestImportDagRulesConfig) -> str | None:
        if "tests" not in file_path.parts:
            return None
        return next(
            (
                facet
                for facet, filename in rules.facet_files.items()
                if file_path.name == filename
            ),
            None,
        )

    @staticmethod
    def _imported_facet(
        imported: str, rules: m.Infra.TestImportDagRulesConfig
    ) -> str | None:
        parts = imported.split(".")
        if not parts or parts[0] != "tests":
            return None
        for part in parts[1:]:
            for facet, filename in rules.facet_files.items():
                if part == filename.removesuffix(".py"):
                    imported_facet: str = t.Infra.STR_ADAPTER.validate_python(facet)
                    return imported_facet
        return None

    @classmethod
    def _edge_violation(
        cls,
        file_path: Path,
        imported: str,
        project_root: Path,
        rules: m.Infra.TestImportDagRulesConfig,
    ) -> str | None:
        relative = file_path.resolve().relative_to(project_root.resolve())
        in_tests = "tests" in relative.parts
        imported_parts = imported.split(".")
        imports_tests = bool(imported_parts) and imported_parts[0] in {
            "tests",
            rules.shared_package,
        }
        if not in_tests and imports_tests:
            return "production code cannot import test infrastructure"
        if project_root.name == "flext-tests" and (
            "fixtures" in relative.parts or file_path.name == "conftest_plugin.py"
        ):
            top = imported_parts[0] if imported_parts else ""
            if top.startswith("flext_") and top not in rules.shared_allowed_imports:
                return "shared test infrastructure cannot import consumer packages"
        if not in_tests:
            return None
        source_facet = cls._facet(file_path, rules)
        imported_facet = cls._imported_facet(imported, rules)
        imports_fixture = any(part in rules.fixture_parts for part in imported_parts)
        imports_test_module = any(
            part.startswith(rules.test_module_prefix) for part in imported_parts
        )
        if source_facet is not None:
            if imported == "tests":
                return "test facets cannot import the tests package root"
            if imports_fixture or imports_test_module:
                return "test facets cannot import fixtures, conftest, or test modules"
            if imported_facet is not None and rules.facet_order.index(
                imported_facet
            ) < rules.facet_order.index(source_facet):
                return "reverse canonical test-facet edge"
        if (
            file_path.name == "__init__.py"
            and relative.parent.name == "tests"
            and (imports_fixture or imports_test_module)
        ):
            return "tests package root cannot import fixtures, conftest, or tests"
        return None

    @override
    def execute(self) -> p.Result[bool]:
        report_result = self.build_report(self.repository_root)
        if report_result.failure:
            return r[bool].fail(
                report_result.error or "test-import-dag validation failed"
            )
        report = report_result.unwrap()
        return r[bool].ok(True) if report.passed else r[bool].fail(report.summary)


__all__: t.StrSequence = ("FlextInfraValidateTestImportDag",)
