"""Project classification and family chain discovery for flext workspace."""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

from flext_infra import c, m, u


class ProjectClassifier:
    """Classify a project by kind and discover MRO family chains."""

    def __init__(self, project_root: Path) -> None:
        """Initialize classifier for the given project root."""
        self._project_root = project_root.resolve()
        self._pyproject_path = self._project_root / c.Infra.Files.PYPROJECT_FILENAME
        self._src_path = self._project_root / c.Infra.Paths.DEFAULT_SRC_DIR

    def classify(self) -> m.Infra.Refactor.ProjectClassification:
        """Return classification and family chains for this project."""
        project_name, dependencies = self._read_project_metadata()
        internal_dependencies = self._internal_dependencies(
            dependencies=dependencies,
            project_name=project_name,
        )
        family_bases, local_facade_classes = self._discover_facade_inheritance()
        family_chains = self._build_confirmed_family_chains(
            internal_dependencies=internal_dependencies,
            family_bases=family_bases,
        )
        project_kind = self._infer_project_kind(
            internal_dependencies=internal_dependencies,
            local_facade_classes=local_facade_classes,
        )
        return m.Infra.Refactor.ProjectClassification(
            project_kind=project_kind,
            family_chains=family_chains,
        )

    def _read_project_metadata(self) -> tuple[str, list[str]]:
        if not self._pyproject_path.is_file():
            return ("", [])
        parsed: dict[str, str | list[str] | dict[str, str | list[str]]] = tomllib.loads(
            self._pyproject_path.read_text(encoding=c.Infra.Encoding.DEFAULT),
        )
        raw_project = parsed.get(c.Infra.Toml.PROJECT)
        if not isinstance(raw_project, dict):
            return ("", [])
        project_name = ""
        raw_name = raw_project.get(c.Infra.Toml.NAME, "")
        if isinstance(raw_name, str):
            project_name = self._normalize_dependency_name(raw_name)
        dependencies: list[str] = []
        raw_dependencies = raw_project.get(c.Infra.Toml.DEPENDENCIES, [])
        if isinstance(raw_dependencies, list):
            for raw_dependency in raw_dependencies:
                dependency_name = self._extract_dependency_name(raw_dependency)
                if dependency_name:
                    dependencies.append(dependency_name)
        return (project_name, dependencies)

    def _internal_dependencies(
        self,
        *,
        dependencies: list[str],
        project_name: str,
    ) -> list[str]:
        internal: list[str] = []
        for dependency in dependencies:
            if not dependency.startswith("flext-"):
                continue
            if dependency == project_name:
                continue
            internal.append(dependency)
        return internal

    def _extract_dependency_name(self, raw_dependency: str) -> str:
        cleaned = raw_dependency.strip().split(";", maxsplit=1)[0].strip()
        if not cleaned:
            return ""
        left_side = cleaned.split(" @ ", maxsplit=1)[0].strip()
        base_token = left_side.split()[0]
        base_token = base_token.split("[", maxsplit=1)[0]
        base_token = re.split(r"[<>=!~]", base_token, maxsplit=1)[0]
        if "/" in base_token:
            path = Path(base_token)
            base_token = path.name
        return self._normalize_dependency_name(base_token)

    def _normalize_dependency_name(self, raw_name: str) -> str:
        normalized = raw_name.strip().lower().replace("_", "-")
        return normalized.strip("./")

    def _discover_facade_inheritance(self) -> tuple[dict[str, set[str]], set[str]]:
        family_bases: dict[str, set[str]] = {
            family: set() for family in c.Infra.Refactor.FAMILY_SUFFIXES
        }
        local_facade_classes: set[str] = set()
        if not self._src_path.is_dir():
            return (family_bases, local_facade_classes)
        for family, suffix in c.Infra.Refactor.FAMILY_SUFFIXES.items():
            file_pattern = c.Infra.Refactor.FAMILY_FILES[family]
            for file_path in self._src_path.rglob(file_pattern):
                class_bases, class_names = self._parse_family_file(file_path, suffix)
                family_bases[family].update(class_bases)
                local_facade_classes.update(class_names)
        return (family_bases, local_facade_classes)

    def _parse_family_file(
        self,
        file_path: Path,
        suffix: str,
    ) -> tuple[set[str], set[str]]:
        tree = u.Infra.parse_module_ast(file_path)
        if tree is None:
            return (set(), set())
        base_names: set[str] = set()
        class_names: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if not node.name.endswith(suffix):
                continue
            class_names.add(node.name)
            for base in node.bases:
                base_name = self._extract_base_name(base)
                if base_name:
                    base_names.add(base_name)
        return (base_names, class_names)

    def _extract_base_name(self, base: ast.expr) -> str:
        if isinstance(base, ast.Name):
            return base.id
        if isinstance(base, ast.Attribute):
            return base.attr
        if isinstance(base, ast.Subscript):
            return self._extract_base_name(base.value)
        return ""

    def _build_confirmed_family_chains(
        self,
        *,
        internal_dependencies: list[str],
        family_bases: dict[str, set[str]],
    ) -> dict[str, list[str]]:
        family_chains: dict[str, list[str]] = {}
        for family, suffix in c.Infra.Refactor.FAMILY_SUFFIXES.items():
            expected_parents = self._expected_parents_for_family(
                family_suffix=suffix,
                internal_dependencies=internal_dependencies,
            )
            confirmed_bases = family_bases.get(family, set())
            confirmed_expected = [
                parent for parent in expected_parents if parent in confirmed_bases
            ]
            extra_confirmed = sorted(
                base for base in confirmed_bases if base not in set(confirmed_expected)
            )
            family_chains[family] = [*confirmed_expected, *extra_confirmed]
        return family_chains

    def _expected_parents_for_family(
        self,
        *,
        family_suffix: str,
        internal_dependencies: list[str],
    ) -> list[str]:
        expected: list[str] = []
        for dependency in internal_dependencies:
            stem = self._dependency_to_class_stem(dependency)
            if not stem:
                continue
            candidate = f"{stem}{family_suffix}"
            if candidate not in expected:
                expected.append(candidate)
        return expected

    def _dependency_to_class_stem(self, dependency: str) -> str:
        normalized = self._normalize_dependency_name(dependency)
        if normalized == c.Infra.Packages.CORE:
            return "Flext"
        if normalized.startswith("flext-"):
            tail = normalized.removeprefix("flext-")
            parts = [part for part in tail.split("-") if part]
            if not parts:
                return ""
            return "Flext" + "".join(part.capitalize() for part in parts)
        parts = [part for part in normalized.split("-") if part]
        if not parts:
            return ""
        return "".join(part.capitalize() for part in parts)

    def _infer_project_kind(
        self,
        *,
        internal_dependencies: list[str],
        local_facade_classes: set[str],
    ) -> str:
        if not internal_dependencies:
            return "core"
        has_domain_dependency = any(
            dependency in c.Infra.Refactor.DOMAIN_PACKAGES
            for dependency in internal_dependencies
        )
        has_platform_dependency = any(
            dependency in c.Infra.Refactor.PLATFORM_PACKAGES
            for dependency in internal_dependencies
        )
        dependency_kind = "app"
        if has_domain_dependency and has_platform_dependency:
            dependency_kind = "integration"
        elif has_domain_dependency:
            dependency_kind = "domain"
        elif has_platform_dependency:
            dependency_kind = "platform"
        has_integration_facade = any(
            class_name.startswith(c.Infra.Refactor.INTEGRATION_CLASS_PREFIXES)
            for class_name in local_facade_classes
        )
        if dependency_kind == "integration" and (not has_integration_facade):
            return "app"
        if (
            dependency_kind == "app"
            and has_domain_dependency
            and has_platform_dependency
            and has_integration_facade
        ):
            return "integration"
        return dependency_kind


__all__ = ["ProjectClassifier"]
