"""Helper-function classification — extracted concern of FlextInfraRefactorViolationAnalyzer."""

from __future__ import annotations

import sys
from collections import Counter
from typing import TYPE_CHECKING

from flext_infra import c, m, u

if TYPE_CHECKING:
    from pathlib import Path

    from flext_infra import t


class FlextInfraRefactorViolationHelperClassifierMixin:
    """Classify loose helper functions into c/m/p/t/u target namespaces.

    Composed into FlextInfraRefactorViolationAnalyzer via inheritance; every
    method is a ``cls``-dispatched classmethod so the facade's ``analyze_files``
    reaches them through FLEXT without hardcoding the facade name.
    """

    @classmethod
    def _analyze_file_helpers(
        cls, *, file_path: Path, content: str
    ) -> m.Infra.HelperFileAnalysis:
        """Analyze file helpers."""
        suggestions: t.MutableSequenceOf[m.Infra.HelperClassification] = []
        totals: Counter[str] = Counter()
        manual_review: t.MutableSequenceOf[m.Infra.HelperClassification] = []
        local_to_import = cls._extract_local_to_import(content)
        for match in c.Infra.FUNCTION_DEF_SIMPLE_RE.finditer(content):
            func_name = match.group(1)
            func_body = cls._extract_function_body(content, match.start())
            used_names = set(c.Infra.IDENTIFIER_RE.findall(func_body))
            dependencies: t.Infra.StrSet = set()
            for name in used_names:
                imported = local_to_import.get(name)
                if imported is not None:
                    dependencies.add(imported)
            has_decorators = bool(c.Infra.DECORATOR_RE.search(func_body))
            matched_categories = cls._match_categories(
                dependencies=dependencies, has_decorators=has_decorators
            )
            classification_category, manual, reason = cls._resolve_category(
                dependencies=dependencies, matched_categories=matched_categories
            )
            namespace_root = c.Infra.NAMESPACE_PREFIXES[classification_category]
            classification = m.Infra.HelperClassification(
                file=str(file_path),
                function=func_name,
                category=classification_category,
                target_namespace=f"{namespace_root}.{func_name}",
                dependencies=sorted(dependencies),
                manual_review=manual,
                review_reason=reason,
            )
            suggestions.append(classification)
            category = classification.category
            totals[category] += 1
            if classification.manual_review:
                manual_review.append(classification)
        return m.Infra.HelperFileAnalysis(
            suggestions=tuple(suggestions),
            totals=dict(totals),
            manual_review=tuple(manual_review),
        )

    @classmethod
    def _match_categories(
        cls, *, dependencies: t.Infra.StrSet, has_decorators: bool
    ) -> t.Infra.StrSet:
        """Match categories."""
        matched: t.Infra.StrSet = set()
        for dependency in dependencies:
            lowered = dependency.lower()
            if any(token in lowered for token in c.Infra.MODEL_TOKENS):
                matched.add("models")
            if any(token in lowered for token in c.Infra.DECORATOR_TOKENS):
                matched.add("decorators")
            if any(token in lowered for token in c.Infra.DISPATCHER_TOKENS):
                matched.add("dispatcher")
        if has_decorators:
            matched.add("decorators")
        return matched

    @classmethod
    def _resolve_category(
        cls, *, dependencies: t.Infra.StrSet, matched_categories: t.Infra.StrSet
    ) -> t.Triple[str, bool, str]:
        """Resolve category."""
        if len(matched_categories) > 1:
            ordered = [
                category
                for category in c.Infra.CLASSIFICATION_PRIORITY
                if category in matched_categories
            ]
            cats = ", ".join(sorted(matched_categories))
            return (ordered[0], True, f"Cross-cutting concerns detected: {cats}")
        if len(matched_categories) == 1:
            category = next(iter(matched_categories))
            return (category, False, "")
        if cls._is_pure_utility_dependencies(dependencies):
            return ("utility", False, "")
        return (
            "utility",
            True,
            "External dependencies outside helper taxonomy; manual review required",
        )

    @classmethod
    def _is_pure_utility_dependencies(cls, dependencies: t.Infra.StrSet) -> bool:
        """Is pure utility dependencies."""
        if not dependencies:
            return True
        for dependency in dependencies:
            root = dependency.split(".", maxsplit=1)[0]
            if root in sys.stdlib_module_names:
                continue
            if root in {"typing", "collections", "dataclasses", "pathlib"}:
                continue
            if root == "builtins":
                continue
            return False
        return True

    @classmethod
    def _extract_local_to_import(cls, content: str) -> t.StrMapping:
        """Extract local-name -> fully-qualified-name mapping from imports."""
        result: t.MutableStrMapping = {}
        for match in c.Infra.COMBINED_IMPORT_RE.finditer(content):
            match match.groups():
                case (str(module_name), str(names_part), _, _):
                    for imported_name, local in u.Infra.parse_import_names(names_part):
                        result[local] = f"{module_name}.{imported_name}"
                case (_, _, str(imported), local_alias):
                    result[local_alias or imported.split(".", maxsplit=1)[0]] = imported
                case _:
                    continue
        return result

    @staticmethod
    def _extract_function_body(content: str, start: int) -> str:
        """Extract function body text from content starting at def position."""
        lines = content[start:].splitlines()
        if not lines:
            return ""
        body_lines: t.MutableSequenceOf[str] = [lines[0]]
        for line in lines[1:]:
            if not line.strip():
                body_lines.append(line)
                continue
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                body_lines.append(line)
            else:
                break
        return "\n".join(body_lines)


__all__: list[str] = ["FlextInfraRefactorViolationHelperClassifierMixin"]
