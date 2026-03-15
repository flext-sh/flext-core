"""Consolidation phase tests for deps modernizer."""

from __future__ import annotations

import tomlkit

from flext_infra import ConsolidateGroupsPhase
from flext_tests import tm


class TestConsolidateGroupsPhase:
    """Tests consolidate groups phase behavior."""

    def test_consolidate_groups_creates_dev_group(self) -> None:
        doc = tomlkit.document()
        project = tomlkit.table()
        optional = tomlkit.table()
        project["optional-dependencies"] = optional
        doc["project"] = project
        changes = ConsolidateGroupsPhase().apply(doc, [])
        tm.that(len(changes) > 0, eq=True)

    def test_consolidate_groups_removes_old_groups(self) -> None:
        doc = tomlkit.parse(
            "[project.optional-dependencies]\n"
            'dev = ["pytest"]\n'
            'docs = ["sphinx"]\n'
            'test = ["coverage"]\n'
        )
        changes = ConsolidateGroupsPhase().apply(doc, ["pytest"])
        tm.that(any("removed" in change for change in changes), eq=True)

    def test_consolidate_groups_merges_poetry_groups(self) -> None:
        doc = tomlkit.document()
        project = tomlkit.table()
        optional = tomlkit.table()
        project["optional-dependencies"] = optional
        doc["project"] = project
        group = tomlkit.table()
        group["dev"] = {"dependencies": {"pytest": "^7.0"}}
        group["docs"] = {"dependencies": {"sphinx": "^4.0"}}
        poetry = tomlkit.table()
        poetry["group"] = group
        tool = tomlkit.table()
        tool["poetry"] = poetry
        doc["tool"] = tool
        changes = ConsolidateGroupsPhase().apply(doc, [])
        tm.that(len(changes) > 0, eq=True)

    def test_consolidate_groups_sets_deptry_config(self) -> None:
        doc = tomlkit.document()
        project = tomlkit.table()
        project["optional-dependencies"] = tomlkit.table()
        doc["project"] = project
        doc["tool"] = tomlkit.table()
        changes = ConsolidateGroupsPhase().apply(doc, [])
        tm.that(any("deptry" in change for change in changes), eq=True)

    def test_consolidate_groups_handles_missing_tables(self) -> None:
        changes = ConsolidateGroupsPhase().apply(tomlkit.document(), [])
        tm.that(len(changes) > 0, eq=True)


def test_consolidate_groups_phase_apply_removes_old_groups() -> None:
    doc = tomlkit.document()
    project = tomlkit.table()
    optional = tomlkit.table()
    optional["dev"] = ["pytest"]
    optional["docs"] = ["sphinx"]
    optional["test"] = ["coverage"]
    project["optional-dependencies"] = optional
    doc["project"] = project
    changes = ConsolidateGroupsPhase().apply(doc, [])
    tm.that(any("optional-dependencies.docs removed" in c for c in changes), eq=True)
    tm.that(any("optional-dependencies.test removed" in c for c in changes), eq=True)


def test_consolidate_groups_phase_apply_with_empty_poetry_group() -> None:
    doc = tomlkit.document()
    project = tomlkit.table()
    project["optional-dependencies"] = tomlkit.table()
    doc["project"] = project
    docs_group = tomlkit.table()
    docs_group["dependencies"] = tomlkit.table()
    group = tomlkit.table()
    group["docs"] = docs_group
    poetry = tomlkit.table()
    poetry["group"] = group
    tool = tomlkit.table()
    tool["poetry"] = poetry
    doc["tool"] = tool
    changes = ConsolidateGroupsPhase().apply(doc, [])
    tm.that(len(changes) > 0, eq=True)
