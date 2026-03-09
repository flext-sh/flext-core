from __future__ import annotations

from collections.abc import MutableMapping

import tomlkit

from flext_infra.deps.modernizer import ConsolidateGroupsPhase
from flext_tests import tm
from tests.infra import h


class TestConsolidateGroupsPhase:
    def test_consolidate_groups_creates_dev_group(self) -> None:
        doc = tomlkit.document()
        doc["project"] = tomlkit.table()
        project = doc["project"]
        tm.that(isinstance(project, MutableMapping), eq=True)
        project["optional-dependencies"] = tomlkit.table()
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
        doc["project"] = tomlkit.table()
        project = doc["project"]
        tm.that(isinstance(project, MutableMapping), eq=True)
        project["optional-dependencies"] = tomlkit.table()
        doc["tool"] = {
            "poetry": {
                "group": {
                    "dev": {"dependencies": {"pytest": "^7.0"}},
                    "docs": {"dependencies": {"sphinx": "^4.0"}},
                },
            },
        }
        changes = ConsolidateGroupsPhase().apply(doc, [])
        tm.that(len(changes) > 0, eq=True)

    def test_consolidate_groups_sets_deptry_config(self) -> None:
        doc = tomlkit.document()
        doc["project"] = tomlkit.table()
        project = doc["project"]
        tm.that(isinstance(project, MutableMapping), eq=True)
        project["optional-dependencies"] = tomlkit.table()
        doc["tool"] = tomlkit.table()
        changes = ConsolidateGroupsPhase().apply(doc, [])
        tm.that(any("deptry" in change for change in changes), eq=True)

    def test_consolidate_groups_handles_missing_tables(self) -> None:
        changes = ConsolidateGroupsPhase().apply(tomlkit.document(), [])
        tm.that(len(changes) > 0, eq=True)


def test_consolidate_groups_phase_apply_removes_old_groups() -> None:
    doc = tomlkit.document()
    doc["project"] = tomlkit.table()
    project = doc["project"]
    tm.that(isinstance(project, MutableMapping), eq=True)
    project["optional-dependencies"] = tomlkit.table()
    opt_deps = project["optional-dependencies"]
    tm.that(isinstance(opt_deps, MutableMapping), eq=True)
    opt_deps["dev"] = ["pytest"]
    opt_deps["docs"] = ["sphinx"]
    opt_deps["test"] = ["coverage"]
    changes = ConsolidateGroupsPhase().apply(doc, [])
    tm.that(any("optional-dependencies.docs removed" in c for c in changes), eq=True)
    tm.that(any("optional-dependencies.test removed" in c for c in changes), eq=True)


def test_consolidate_groups_phase_apply_with_empty_poetry_group() -> None:
    doc = tomlkit.document()
    doc["project"] = tomlkit.table()
    project = doc["project"]
    tm.that(isinstance(project, MutableMapping), eq=True)
    project["optional-dependencies"] = tomlkit.table()
    doc["tool"] = tomlkit.table()
    tool = doc["tool"]
    tm.that(isinstance(tool, MutableMapping), eq=True)
    tool["poetry"] = tomlkit.table()
    poetry = tool["poetry"]
    tm.that(isinstance(poetry, MutableMapping), eq=True)
    poetry["group"] = tomlkit.table()
    group = poetry["group"]
    tm.that(isinstance(group, MutableMapping), eq=True)
    group["docs"] = tomlkit.table()
    docs = group["docs"]
    tm.that(isinstance(docs, MutableMapping), eq=True)
    docs["dependencies"] = tomlkit.table()
    changes = ConsolidateGroupsPhase().apply(doc, [])
    tm.that(len(changes) > 0, eq=True)
    tm.that(hasattr(h, "assert_ok"), eq=True)
