"""Project metadata utility tests.

Covers the surviving public project-metadata utilities: ``u.lazy_alias_suffixes``
(installed-distribution lazy-alias suffix table) and the ``m.PyprojectDocument``
ingress model (nested PEP 621 ``project`` + ``[tool.flext]`` contract).
"""

from __future__ import annotations

from flext_tests import tm
from tests import m, u


class TestsFlextCoreUtilitiesProjectMetadata:
    """Verify project metadata utilities."""

    def test_lazy_alias_suffixes_returns_tuple_for_installed_distribution(self) -> None:
        tm.that(u.lazy_alias_suffixes("flext-core"), is_=tuple)

    def test_lazy_alias_suffixes_is_empty_for_unknown_distribution(self) -> None:
        tm.that(u.lazy_alias_suffixes("nonexistent-distribution-xyz"), eq=())

    def test_pyproject_document_parses_nested_project_and_tool(self) -> None:
        doc = m.PyprojectDocument.model_validate({
            "project": {"name": "flext-ldif", "version": "1.0.0"},
            "tool": {"flext": {"workspace": {"attached": True}}},
        })
        dumped = doc.model_dump()
        tm.that(dumped["project"]["name"], eq="flext-ldif")
        tm.that(dumped["project"]["version"], eq="1.0.0")
        tm.that(dumped["tool"]["flext"]["workspace"]["attached"], eq=True)

    def test_pyproject_document_populates_tool_defaults_when_absent(self) -> None:
        doc = m.PyprojectDocument.model_validate({
            "project": {"name": "flext-ldif", "version": "1.0.0"}
        })
        tm.that(doc.tool is not None, eq=True)
        tm.that(doc.tool.flext is not None, eq=True)

    def test_pyproject_document_model_dump_roundtrips(self) -> None:
        doc = m.PyprojectDocument.model_validate({
            "project": {"name": "flext-ldif", "version": "1.0.0"}
        })
        rebuilt = m.PyprojectDocument.model_validate(doc.model_dump())
        tm.that(rebuilt.model_dump()["project"]["name"], eq="flext-ldif")
